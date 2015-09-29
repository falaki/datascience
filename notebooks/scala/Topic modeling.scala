// Databricks notebook source exported at Tue, 29 Sep 2015 22:12:06 UTC
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.mllib.clustering.{LDA, OnlineLDAOptimizer, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vector

import sqlContext.implicits._

val numTopics: Int = 100
val maxIterations: Int = 20
val vocabSize: Int = 5000

// COMMAND ----------

val tokens = new RegexTokenizer()
  .setGaps(false)
  .setPattern("\\p{L}+")
  .setInputCol("text")
  .setOutputCol("words")
  .transform(data)

// COMMAND ----------

val stopwords: Array[String] = sc.textFile("/mnt/hossein/text/stopwords.txt").flatMap(_.stripMargin.split("\\s+")).collect ++ Array("template", "taxobox","text", "stub", "template:Taxobox", "template:cite", "de")

// COMMAND ----------

val filteredTokens = new StopWordsRemover()
  .setStopWords(stopwords)
  .setCaseSensitive(false)
  .setInputCol("words")
  .setOutputCol("filtered")
  .transform(tokens)

// COMMAND ----------

val cvModel = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("features")
  .setVocabSize(vocabSize)
  .fit(filteredTokens)

// COMMAND ----------

val countVectors = cvModel.transform(filteredTokens)
  .select("id", "features")
  .map { case r => (r.getAs[String](0).toLong, r.getAs[Vector](1)) }
  .cache()

// COMMAND ----------

val lda = new LDA()
   .setDocConcentration(-1)
  .setTopicConcentration(-1)
  .setK(numTopics)
  .setMaxIterations(maxIterations)

// COMMAND ----------

val ldaModel: DistributedLDAModel = lda.run(countVectors).asInstanceOf[DistributedLDAModel]

// COMMAND ----------

val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 5)
val vocabArray = cvModel.vocabulary
val topics = topicIndices.map { case (terms, termWeights) =>
  terms.map(vocabArray(_)).zip(termWeights)
}
println(s"$numTopics topics:")
topics.zipWithIndex.foreach { case (topic, i) =>
  println(s"Topic $i")
  topic.foreach { case (term, weight) => println(s"$term\t$weight") }
  println(s"-" * 20)
}

// COMMAND ----------

