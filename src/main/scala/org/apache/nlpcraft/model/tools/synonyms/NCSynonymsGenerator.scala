/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.nlpcraft.model.tools.synonyms

import java.util.concurrent.{CopyOnWriteArrayList, CountDownLatch, TimeUnit}

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.apache.http.HttpResponse
import org.apache.http.client.ResponseHandler
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}
import org.apache.http.util.EntityUtils
import org.apache.nlpcraft.common.ascii.NCAsciiTable
import org.apache.nlpcraft.common.makro.NCMacroParser
import org.apache.nlpcraft.common.nlp.core.NCNlpPorterStemmer
import org.apache.nlpcraft.common.util.NCUtils
import org.apache.nlpcraft.model.NCModelFileAdapter

import scala.collection.JavaConverters._
import scala.collection._

/**
  * // TODO: all string fields
  *
  * TODO:
  * @param url
  * @param modelPath
  * @param minFactor
  */
case class NCSynonymsGenerator(url: String, modelPath: String, minFactor: Double) {
    /**
      * Suggestion data holder.
      *
      * @param word Word
      * @param bert Bert factor.
      * @param normalized Normalized bert factor.
      * @param ftext FText factor.
      * @param score Calculated summary factor: normalized * weight1 + ftext * weight2 (weights values are 1 currently)
      */
    case class Suggestion(word: String, bert: String, normalized: String, ftext: String, score: String)
    case class Request(sentence: String, simple: Boolean)
    case class Response(data: java.util.ArrayList[Suggestion])

    private val GSON = new Gson
    private val TYPE_RESP = new TypeToken[Response]() {}.getType
    private val SEPARATORS = Seq('?', ',', '.', '-', '!')

    private val HANDLER = new ResponseHandler[Seq[Suggestion]]() {
        override def handleResponse(resp: HttpResponse): Seq[Suggestion] = {
            val code = resp.getStatusLine.getStatusCode
            val e = resp.getEntity

            val js = if (e != null) EntityUtils.toString(e) else null

            if (js == null)
                throw new RuntimeException(s"Unexpected empty response [code=$code]")

            code match {
                case 200 ⇒
                    val data: Response = GSON.fromJson(js, TYPE_RESP)

                    data.data.asScala

                case 400 ⇒ throw new RuntimeException(js)
                case _ ⇒ throw new RuntimeException(s"Unexpected response [code=$code, text=$js]")
            }
        }
    }

    private def split(s: String): Seq[String] = s.split(" ").toSeq.map(_.trim).filter(_.nonEmpty)

    private def toStem(s: String): String = split(s).map(NCNlpPorterStemmer.stem).mkString(" ")
    private def toStemWord(s: String): String = NCNlpPorterStemmer.stem(s)

    private def ask(client: CloseableHttpClient, sen: String): Seq[Suggestion] = {
        val post = new HttpPost(url)

        post.setHeader("Content-Type", "application/json")
        post.setEntity(new StringEntity(GSON.toJson(Request(sen, simple = false)), "UTF-8"))

        try
            client.execute(post, HANDLER)
        finally
            post.releaseConnection()
    }

    def process(): Unit = {
        val mdl = new NCModelFileAdapter(modelPath) {}

        val parser = new NCMacroParser()

        if (mdl.getMacros != null)
            mdl.getMacros.asScala.foreach { case (name, str) ⇒ parser.addMacro(name, str) }

        val client = HttpClients.createDefault

        case class Word(word: String, stem: String) {
            require(!word.contains(" "), s"Word cannot contains spaces: $word")
            require(
                word.forall(ch ⇒
                    ch.isLetterOrDigit ||
                    ch == '\'' ||
                    SEPARATORS.contains(ch)
                ),
                s"Unsupported symbols: $word"
            )
        }

        val examples =
            mdl.getExamples.asScala.
                map(s ⇒ SEPARATORS.foldLeft(s)((s, ch) ⇒ s.replaceAll(s"\\$ch", s" $ch "))).
                map(split).
                map(_.map(p ⇒ Word(p, toStemWord(p)))).
                toSeq

        val elemSyns =
            mdl.getElements.asScala.map(e ⇒ e.getId → e.getSynonyms.asScala.flatMap(parser.expand)).
                map { case (id, seq) ⇒ id → seq.map(txt ⇒ split(txt).map(p ⇒ Word(p, toStemWord(p))))}.toMap


        val allSens: Map[String, Seq[String]] =
            elemSyns.map {
                case (elemId, elemSyns) ⇒
                    val elemSingleSyns = elemSyns.filter(_.size == 1).map(_.head)
                    val elemStems = elemSingleSyns.map(_.stem)

                    val hs =
                        examples.flatMap(example ⇒ {
                            val exStems = example.map(_.stem)
                            val idxs = exStems.flatMap(s ⇒ if (elemStems.contains(s)) Some(exStems.indexOf(s)) else None)

                            if (idxs.nonEmpty)
                                elemSingleSyns.map(_.word).flatMap(syn ⇒
                                    idxs.map(idx ⇒
                                        example.
                                        zipWithIndex.map { case (w, i1) ⇒ if (idxs.contains(i1)) syn else w.word }.
                                        zipWithIndex.map { case (s, i2) ⇒ if (i2 == idx) s"$s#" else s}.
                                        mkString(" ")
                                    )
                                )
                            else
                                Seq.empty
                        })

                    elemId → hs
            }.filter(_._2.nonEmpty)

        val cache = new java.util.concurrent.ConcurrentHashMap[String, Seq[Suggestion]] ()
        val allSuggs = new java.util.concurrent.ConcurrentHashMap[String, java.util.List[Suggestion]] ()

        val cdl = new CountDownLatch(allSens.map { case (_, seq) ⇒ seq.size }.sum)

        for ((elemId, sens) ← allSens; sen ← sens)
            NCUtils.asFuture(
                _ ⇒ {
                    allSuggs.computeIfAbsent(elemId, (_: String) ⇒ new CopyOnWriteArrayList[Suggestion]()).
                        addAll(cache.computeIfAbsent(sen, (_: String) ⇒ ask(client, sen)).asJava)
                },
                (e: Throwable) ⇒ {
                    e.printStackTrace()

                    cdl.countDown()
                },
                (_: Boolean) ⇒ cdl.countDown()
            )

        cdl.await(Long.MaxValue, TimeUnit.MILLISECONDS)

        val allSynsStems = elemSyns.flatMap(_._2).flatten.map(_.stem).toSet

        val table = NCAsciiTable()

        table #= ("Element", "Suggestions")

        allSuggs.asScala.map { case (id, elemSuggs) ⇒ id → elemSuggs.asScala}.foreach { case (elemId, elemSuggs) ⇒
            elemSuggs.
                map(sugg ⇒ (sugg, toStem(sugg.word))).
                groupBy { case (_, stem) ⇒ stem }.
                filter { case (stem, _) ⇒ !allSynsStems.contains(stem) }.
                map { case (_, group) ⇒
                    val seq = group.map { case (sugg, _) ⇒ sugg }.sortBy(-_.score.toDouble)

                    // Drops repeated.
                    (seq.head, seq.length)
                }.
                // TODO: develop more intelligent sorting.
                toSeq.sortBy { case (sugg, cnt) ⇒ (-cnt , -sugg.score.toDouble) }.
                zipWithIndex.
                foreach { case ((sugg, cnt), sugIdx) ⇒
                    table += (
                        if (sugIdx == 0) elemId else " ",
                        s"${sugg.word} " +
                            s"[count=$cnt, " +
                            s"bert=${sugg.bert}, " +
                            s"ftext=${sugg.ftext}, " +
                            s"norm=${sugg.normalized}, " +
                            s"score=${sugg.score}" +
                            s"]"
                    )
                }
        }

        table.render()
    }
}

object NCSynonymsGeneratorRunner extends App {
    NCSynonymsGenerator(
        url = "http://localhost:5000",
        modelPath = "src/main/scala/org/apache/nlpcraft/examples/weather/weather_model.json",
        minFactor = 0
    ).process()
}
