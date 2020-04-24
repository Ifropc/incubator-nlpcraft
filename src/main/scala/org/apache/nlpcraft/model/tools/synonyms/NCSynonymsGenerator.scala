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

import scala.collection.JavaConverters._

object NCSynonymsGenerator extends App {
    // TODO: all string fields
    case class Holder(word: String, bert: String, normalized: String, ftext: String, score: String) {
        override def toString: String = s"$word [bert=$bert, ftext=$ftext, normalized=$normalized, score=$score]"
    }
    case class Request(sentence: String, simple: Boolean)
    case class Response(data: java.util.ArrayList[Holder])

    private val GSON = new Gson
    private val TYPE_RESP: Type = new TypeToken[Response]() {}.getType

    private def split(s: String): Seq[String] = s.split(" ").toSeq.map(_.trim).filter(_.nonEmpty)

    private def ask(client: CloseableHttpClient, url: String, words: Seq[String], idx: Int, minFactor: Double): Seq[Holder]= {
        val sen = words.zipWithIndex.map { case (word, wordIdx) ⇒ if (wordIdx == idx) s"$word#" else word }.mkString(" ")

        val post = new HttpPost(url)

        post.setHeader("Content-Type", "application/json")
        post.setEntity(new StringEntity(GSON.toJson(Request(sen, simple = false)), "UTF-8"))

        val h = new ResponseHandler[Seq[Holder]]() {
            override def handleResponse(resp: HttpResponse): Seq[Holder] = {
                val code = resp.getStatusLine.getStatusCode
                val e = resp.getEntity

                val js = if (e != null) EntityUtils.toString(e) else null

                if (js == null)
                    throw new RuntimeException(s"Unexpected empty response [code=$code]")

                code match {
                    case 200 ⇒
                        // TODO: add filter by minFactor.
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

    private def process(mdlPath: String, url: String): Unit = {
        val mdl = new NCModelFileAdapter(mdlPath) {
            // No-op.
        }

        val parser = new NCMacroParser()

        if (mdl.getMacros != null)
            mdl.getMacros.asScala.foreach { case (name, str) ⇒ parser.addMacro(name, str) }

        val table = NCAsciiTable()
        val client: CloseableHttpClient = HttpClients.createDefault

        table #= ("Single synonym", "Suggestions")

        val examples: Set[(Seq[String], Seq[String])] =
            mdl.getExamples.asScala.
                // TODO: Is it enough?
                map(_.replaceAll("\\?", "")).
                map(_.replaceAll("\\.", "")).
                map(_.replaceAll("!", "")).
                map(split).map(p ⇒ p → p.map(NCNlpPorterStemmer.stem)).toSet

        val suggestions =
            mdl.getElements.asScala.flatMap(e ⇒ {
                val elemSyns = e.getSynonyms.asScala.flatMap(p ⇒ parser.expand(p)).map(s ⇒ s → split(s)).toMap

                elemSyns.filter(_._2.length == 1).
                    map(_._2.head).
                    map(p ⇒ p → NCNlpPorterStemmer.stem(p)).
                    flatMap { case (syn, synStem) ⇒
                        val suggestions: Set[Holder] =
                            examples.filter(_._2.contains(synStem)).flatMap { case (eWords, eStems) ⇒
                                val idx = eStems.indexOf(synStem)

                                require(idx >= 0)

                                ask(client, url, eWords, idx, 0.0)
                            }.filter(p ⇒ !elemSyns.contains(p.word))

                        if (suggestions.nonEmpty) Some(syn → suggestions) else None
                    }.toMap
            }).toMap

        val n = suggestions.size

        suggestions.zipWithIndex.map { case ((syn, hs), idx) ⇒
            // TODO: sort
            hs.toSeq.sortBy(_.score.toDouble).reverse.foreach(h ⇒ table += (syn, h))

            if (idx != n - 1)
                table += ("-------", "-------")
        }

        table.render()
    }

    process(
        "src/main/scala/org/apache/nlpcraft/examples/weather/weather_model.json",
        "http://localhost:5000"
    )
}
