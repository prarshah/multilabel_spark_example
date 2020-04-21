import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Logger, Level}
object poc3 {

  def main(args: Array[String]): Unit = {
    val logger = Logger.getRootLogger
    logger.setLevel(Level.ERROR)

    val sparkConf = new SparkConf()
      .setAppName("H2OAutoML_poc_2019")
      .setMaster("local[*]")
      .set("spark.driver.host", "localhost")

    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    import org.apache.spark.mllib.evaluation.MultilabelMetrics
    import org.apache.spark.rdd.RDD
    import sparkSession.implicits._

    val labelColName = "trueLabels"
    val predictionColname = "prediction"
    val scoreAndLabels: RDD[(Array[Double], Array[Double])] =
      sparkSession.sparkContext.parallelize(
        Seq(
          (Array(0.0, 1.0), Array(0.0, 2.0)),
          (Array(0.0, 2.0), Array(0.0, 1.0)),
          (Array.empty[Double], Array(0.0)),
          (Array(2.0), Array(2.0)),
          (Array(2.0, 0.0), Array(2.0, 0.0)),
          (Array(0.0, 1.0, 2.0), Array(0.0, 1.0)),
          (Array(1.0), Array(1.0, 2.0))
        ),
        2
      )
    val df = scoreAndLabels.toDF(predictionColname, labelColName)
    //  poc starts here

    val metric: Double = new MultilabelClassificationEvaluator()
      .setLabelCol(labelColName)
      .setPredictionCol(predictionColname)
      .setMetricName("hammingLoss")
      .evaluate(df)
    println("\n\n" + s"Metric = ${metric}")

    val metrics = new MultilabelMetrics(scoreAndLabels)

    // Summary stats
    println("\n\n" + s"Recall = ${metrics.recall}")
    println("\n\n" + s"Precision = ${metrics.precision}")
    println("\n\n" + s"F1 measure = ${metrics.f1Measure}")
    println("\n\n" + s"Accuracy = ${metrics.accuracy}")

    // Individual label stats
    metrics.labels.foreach(
      label =>
        println(
          "\n\n" + s"Class $label precision = ${metrics.precision(label)}"
      )
    )
    metrics.labels.foreach(
      label =>
        println("\n\n" + s"Class $label recall = ${metrics.recall(label)}")
    )
    metrics.labels.foreach(
      label =>
        println("\n\n" + s"Class $label F1-score = ${metrics.f1Measure(label)}")
    )

    // Micro stats
    println("\n\n" + s"Micro recall = ${metrics.microRecall}")
    println("\n\n" + s"Micro precision = ${metrics.microPrecision}")
    println("\n\n" + s"Micro F1 measure = ${metrics.microF1Measure}")

    // Hamming loss
    println("\n\n" + s"Hamming loss = ${metrics.hammingLoss}")

    // Subset accuracy
    println("\n\n" + s"Subset accuracy = ${metrics.subsetAccuracy}")
  }
}
