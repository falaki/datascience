// Databricks notebook source exported at Tue, 20 Oct 2015 00:34:05 UTC
import org.apache.spark.ml.feature.{HashingTF, CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.mllib.clustering.{LDA, OnlineLDAOptimizer, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.LogisticRegressionModel


// COMMAND ----------

// MAGIC %md ### Fetching the training data we just created, as a DataFrame

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

val featurizer = new HashingTF()
  .setNumFeatures(10000)
  .setInputCol("filtered")
  .setOutputCol("features")


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

val lrModel = pipeline.fit(data)

// COMMAND ----------

