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

import java.lang.reflect.Type

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
import org.apache.nlpcraft.model.NCModelFileAdapter

import scala.collection._
import scala.collection.JavaConverters._

case class NCSynonymsGenerator(url: String, modelPath: String, minFactor: Double) {
    // TODO: all string fields
    // normalized  - normalized bert value.
    // score = normalized * weight + ftext * weight
    // both `weights` = 1
    case class Suggestion(word: String, bert: String, normalized: String, ftext: String, score: String) {
        override def toString: String = s"$word [bert=$bert, ftext=$ftext, normalized=$normalized, score=$score]"
    }

    case class Request(sentence: String, simple: Boolean)

    case class Response(data: java.util.ArrayList[Suggestion])

    private val GSON = new Gson
    private val TYPE_RESP: Type = new TypeToken[Response]() {}.getType

    private def split(s: String): Seq[String] = s.split(" ").toSeq.map(_.trim).filter(_.nonEmpty)

    private def ask(client: CloseableHttpClient, sen: String): Seq[Suggestion] = {
        val post = new HttpPost(url)

        post.setHeader("Content-Type", "application/json")
        post.setEntity(new StringEntity(GSON.toJson(Request(sen, simple = false)), "UTF-8"))

        val h = new ResponseHandler[Seq[Suggestion]]() {
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

        try
            client.execute(post, h)
        finally
            post.releaseConnection()
    }

    def process(): Unit = {
        val mdl = new NCModelFileAdapter(modelPath) {}

        val parser = new NCMacroParser()

        if (mdl.getMacros != null)
            mdl.getMacros.asScala.foreach { case (name, str) ⇒ parser.addMacro(name, str) }

        val client = HttpClients.createDefault

        case class Word(word: String) {
            val stem: String = NCNlpPorterStemmer.stem(word)
        }

        val examples: Seq[Seq[Word]] =
            mdl.getExamples.asScala.
                // TODO: Is it enough?
                map(_.replaceAll("\\?", " ?")).
                map(_.replaceAll("\\.", " .")).
                map(_.replaceAll(",", " ,")).
                map(_.replaceAll("!", " !")).
                map(split).
                map(_.map(Word)).
                toSeq

        val elemSyns = mdl.getElements.asScala.map(e ⇒ e.getId → e.getSynonyms.asScala.flatMap(parser.expand)).toMap

        val cache = mutable.HashMap.empty[String, Seq[Suggestion]]

        val allSuggs =
            elemSyns.map {
                case (elemId, elemSyns) ⇒
                    val stemsSyns: Seq[(String, String)] =
                        elemSyns.
                            map(text ⇒ text → split(text).map(Word)).
                            filter { case( _, words) ⇒ words.size == 1 }.
                            map { case(text, words) ⇒ words.head.stem → text }

                    val hs: Seq[Suggestion] =
                        examples.flatMap(exWords ⇒ {
                            val exStems = exWords.map(_.stem)

                            val idxs =
                                exStems.flatMap(stem ⇒
                                    stemsSyns.find(_._1 == stem) match {
                                        case Some(p) ⇒ Some(exStems.indexOf(p._1))
                                        case None ⇒ None
                                    }
                                )

                            if (idxs.nonEmpty)
                                stemsSyns.map(_._2).flatMap(syn ⇒ {
                                    val wordsTxt =
                                        exWords.zipWithIndex.map { case (word, idx) ⇒ if (idxs.contains(idx)) syn else word.word }

                                    idxs.flatMap(idx ⇒ {
                                        val sen =
                                            wordsTxt.zipWithIndex.map {
                                                case (word, wordIdx) ⇒ if (wordIdx == idx) s"$word#" else word
                                            }.mkString(" ")

                                        cache.get(sen) match {
                                            case Some(res) ⇒ res
                                            case None ⇒
                                                val res: Seq[Suggestion] = ask(client, sen).filter(_.score.toDouble >= minFactor)

                                                cache += sen → res

                                                res
                                        }
                                    })
                                })
                            else
                                Seq.empty
                        })

                    elemId → hs
            }.filter(_._2.nonEmpty)

        val allSyns = elemSyns.flatMap(_._2).toSet

        val table = NCAsciiTable()

        table #= ("Element", "Suggestions")

        allSuggs.foreach { case (elemId, elemSuggs) ⇒
            elemSuggs.
                groupBy(_.word).
                map { case (_, group) ⇒ group.sortBy(_.score.toDouble).reverse.head }. // Drops repeated.
                toSeq.sortBy(_.score.toDouble).reverse.
                filter(p ⇒ !allSyns.contains(p.word)). // TODO: drop by stem, not by word as is
                zipWithIndex.
                foreach { case (sugg, sugIdx) ⇒ table += (if (sugIdx == 0) elemId else " ", sugg) }
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
