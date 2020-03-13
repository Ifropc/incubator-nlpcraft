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

package org.nlpcraft

import org.nlpcraft.common.NCService

/**
  * NLPCraft lifecycle components test helper.
  */
trait NCTestStartStop {
    private val srvs = scala.collection.mutable.Buffer.empty[NCService]

    /**
     * Starts services.
     *
     * @param srvs Services.
     */
    def startServices(srvs: NCService*): Unit =
        srvs.synchronized {
            srvs.foreach(s ⇒ this.srvs += s)
        }

    /**
     * Stops started services.
     */
    def stopServices(): Unit = {
        srvs.synchronized {
            srvs.reverse.foreach(p ⇒ {
                try
                    p.stop()
                catch {
                    case e: Exception ⇒ e.printStackTrace()
                }
            })
            
            srvs.clear()
        }
    }
}