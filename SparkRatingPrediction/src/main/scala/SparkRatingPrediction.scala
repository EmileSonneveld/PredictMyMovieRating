import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.ScalaObjectMapper
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression}
import org.apache.spark.ml.feature.{Binarizer, VectorAssembler}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Row, SparkSession}

import scala.io.Source

object SparkRatingPrediction extends App {
  val spark = SparkSession.builder()
    .master("local[1]")
    .appName("SparkRatingPrediction")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  def loadMetaCriticDF() = {
    val imdbToMetacriticPath = "../PlaywrightScrapeImdb/imdb_to_metacritic.json"
    val json = Source.fromFile(imdbToMetacriticPath)

    val mapper = new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    val parsedJson = mapper.readValue[Map[String, Object]](json.reader())

    var resultSeq = Seq[(String, Int)]()

    def appendNonEmptyToResult(a: Any): Unit = a match {
      case (_: String, _: String) => println("ignore: " + a)
      case (x1: String, x2: Int) => resultSeq = resultSeq :+ (x1, x2)
    }

    parsedJson.foreach(appendNonEmptyToResult)
    spark.createDataFrame(resultSeq).toDF("Const", "metaScore")
  }

  def splitGenreTags(genresultDF: org.apache.spark.sql.DataFrame) = {
    // Nice to have: Determine tag columns in runtime
    var sequence = Seq[(String, Int, Int, Int, Int, Int, Int, Int, Int,
      Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)]()
    genresultDF.select("Const", "Genres").collect().foreach({
      case Row(const: String, genresStr: String) =>
        var tuple = (const, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        genresStr.split(", ").foreach({
          case "Action" => tuple = tuple.copy(_2 = 1)
          case "Adventure" => tuple = tuple.copy(_3 = 1)
          case "Animation" => tuple = tuple.copy(_4 = 1)
          case "Biography" => tuple = tuple.copy(_5 = 1)
          case "Comedy" => tuple = tuple.copy(_6 = 1)
          case "Crime" => tuple = tuple.copy(_7 = 1)
          case "Documentary" => tuple = tuple.copy(_8 = 1)
          case "Drama" => tuple = tuple.copy(_9 = 1)
          case "Family" => tuple = tuple.copy(_10 = 1)
          case "Fantasy" => tuple = tuple.copy(_11 = 1)
          case "History" => tuple = tuple.copy(_12 = 1)
          case "Horror" => tuple = tuple.copy(_13 = 1)
          case "Musical" => tuple = tuple.copy(_14 = 1)
          case "Mystery" => tuple = tuple.copy(_15 = 1)
          case "Romance" => tuple = tuple.copy(_16 = 1)
          case "Sci-Fi" => tuple = tuple.copy(_17 = 1)
          case "Short" => tuple = tuple.copy(_18 = 1)
          case "Sport" => tuple = tuple.copy(_19 = 1)
          case "Thriller" => tuple = tuple.copy(_20 = 1)
          case "War" => tuple = tuple.copy(_21 = 1)
          case "Western" => tuple = tuple.copy(_22 = 1)
          case unrecognisedGenre => println("genre not recognized: ", unrecognisedGenre)
        })
        sequence = sequence :+ tuple
    })

    spark.createDataFrame(sequence).toDF(
      "Const",
      "genre-Action",
      "genre-Adventure",
      "genre-Animation",
      "genre-Biography",
      "genre-Comedy",
      "genre-Crime",
      "genre-Documentary",
      "genre-Drama",
      "genre-Family",
      "genre-Fantasy",
      "genre-History",
      "genre-Horror",
      "genre-Musical",
      "genre-Mystery",
      "genre-Romance",
      "genre-Sci-Fi",
      "genre-Short",
      "genre-Sport",
      "genre-Thriller",
      "genre-War",
      "genre-Western",
    )
  }

  var moviesDF = spark.read.format("com.databricks.spark.csv")
    .option("delimiter", ",")
    .option("header", "false")
    .load("../ratings.csv")
    .toDF("Const", "Your Rating", "Date Rated", "Title", "URL", "Title Type", "IMDb Rating",
      "Runtime (mins)", "Year", "Genres", "Num Votes", "Release Date", "Directors")
    .withColumn("Your Rating", col("Your Rating").cast("double"))
    .withColumn("IMDb Rating", col("IMDb Rating").cast("double"))
    .withColumn("Runtime (mins)", col("Runtime (mins)").cast("double"))
    .withColumn("Year", col("Year").cast("double"))

  val binarizer: Binarizer = new Binarizer()
    .setInputCol("Your Rating")
    .setOutputCol("label")
    .setThreshold(7.99) // Subjective: if I gave a rating of 8 or more, I liked the movie.
  moviesDF = binarizer.transform(moviesDF)

  // For non-binary classification:
  // moviesDF = moviesDF.withColumnRenamed("Your Rating", "label")

  val goodMoviesCount = moviesDF.filter(r => r.getAs[Double]("label") == 1).count

  val metaCriticDF = loadMetaCriticDF()
  metaCriticDF.show()
  moviesDF = moviesDF.join(metaCriticDF, Seq("Const"), "inner")

  val genreTags = splitGenreTags(moviesDF)
  moviesDF = moviesDF.join(genreTags, Seq("Const"), "inner")
  moviesDF.show()
  println("good movies: " + goodMoviesCount + "/" + moviesDF.count)

  val assembler = new VectorAssembler()
    .setInputCols(Array(
      "IMDb Rating",
      "Runtime (mins)",
      "Year",
      "metaScore",
      "genre-Action",
      "genre-Adventure",
      "genre-Animation",
      "genre-Biography",
      "genre-Comedy",
      "genre-Crime",
      "genre-Documentary",
      "genre-Drama",
      "genre-Family",
      "genre-Fantasy",
      "genre-History",
      "genre-Horror",
      "genre-Musical",
      "genre-Mystery",
      "genre-Romance",
      "genre-Sci-Fi",
      "genre-Short",
      "genre-Sport",
      "genre-Thriller",
      "genre-War",
      "genre-Western",
    ))
    .setOutputCol("features")

  var moviesVectorizedDF = assembler.transform(moviesDF)
  moviesVectorizedDF.show()
  moviesVectorizedDF = moviesVectorizedDF.select("Const", "label", "features")
  println("moviesVectorizedDF.count: " + moviesVectorizedDF.count)

  val bootstrapBuckets = 1
  //val bootstrapBuckets = 100 // Uncomment to see how stable the estimators get trained.
  (1 to bootstrapBuckets) foreach { _ =>
    // randomSplit can cause problems in spark 3.0.0-preview around 'nanSafeCompareDoubles'
    val Array(trainingDF, testDF) = moviesVectorizedDF.randomSplit(Array(0.7, 0.3)) // , seed = 1234
    println("trainingDF.count: " + trainingDF.count)

    val ratingClassifier = new DecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxDepth(2)
      .setMaxBins(20)

    // ratingClassifier will use columns 'label' and 'features' by default
    //    val ratingClassifier = new LogisticRegression()
    //      .setMaxIter(10)
    //      .setRegParam(0.01)

    val ratingModel = ratingClassifier.fit(trainingDF)
    println("ratingModel.toDebugString:\n" + ratingModel.toDebugString)
    // Since ratingModel is a Model (i.e., a Transformer produced by an Estimator),
    // we can view the parameters it used during fit().
    println("ratingModel was fit using parameters: " + ratingModel.parent.extractParamMap)
    //  println("weightedPrecision (during training)", ratingModel.summary.weightedPrecision)
    //  println("weightedRecall (during training)", ratingModel.summary.weightedRecall)

    // Make predictions on test data using the Transformer.transform() method.
    // this will only use the 'features' column.
    val predictDF = ratingModel.transform(testDF)
      .select("features", "label", "probability", "prediction")
    predictDF.show(10000) // Shows all results

    // Calculate precision and recall, the old fasioned way:
    val truePositive = predictDF.filter(
      r => (r.getAs[Double]("prediction") == 1) && (r.getAs[Double]("label") == 1)).count
    val allPositivePredictions = predictDF.filter(r => r.getAs[Double]("prediction") == 1).count
    val allPositiveLabels = predictDF.filter(r => r.getAs[Double]("label") == 1).count
    val precision = truePositive.toDouble / allPositivePredictions.toDouble
    val recall = truePositive.toDouble / allPositiveLabels.toDouble
    val f1Score = 2 * precision * recall / (precision + recall)
    println("testDF.count: " + testDF.count)
    println("truePositive: " + truePositive)
    println("allPositivePredictions: " + allPositivePredictions)
    println("allPositiveLabels: " + allPositiveLabels)
    println("precision: " + precision)
    println("recall: " + recall)
    println("f1Score: " + f1Score)

    val resultDF = predictDF.select("label", "prediction")
    val scoreAndLabels = resultDF.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    println("metrics.areaUnderPR(): " + metrics.areaUnderPR())

    println("done")
  }
}
