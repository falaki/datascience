// Databricks notebook source exported at Tue, 27 Oct 2015 14:34:02 UTC
val data = table("tweetData").unionAll(table("reviewData"))

// COMMAND ----------

import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline


// COMMAND ----------

// MAGIC %md ![pipeline](https://databricks-hossein.s3.amazonaws.com/Plots/pipeline.png)

// COMMAND ----------

val trainingData = data.withColumn("label", when(data("isHappy"), 1.0D).otherwise(0.0D))

// COMMAND ----------

// MAGIC %md ### Running the pipeline

// COMMAND ----------

val tokenizer = new RegexTokenizer()
  .setGaps(false)
  .setPattern("\\p{L}+")
  .setInputCol("text")
  .setOutputCol("words")

// COMMAND ----------

val stopwords: Array[String] = sc.textFile("/mnt/hossein/text/stopwords.txt").flatMap(_.stripMargin.split("\\s+")).collect ++ Array("rt")

// COMMAND ----------

val filterer = new StopWordsRemover()
  .setStopWords(stopwords)
  .setCaseSensitive(false)
  .setInputCol("words")
  .setOutputCol("filtered")

// COMMAND ----------

val countVectorizer = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("features")

// COMMAND ----------

val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.2)
  .setElasticNetParam(0.0)

// COMMAND ----------

val pipeline = new Pipeline().setStages(Array(tokenizer, filterer, countVectorizer, lr))

// COMMAND ----------

val lrModel = pipeline.fit(trainingData)

// COMMAND ----------

