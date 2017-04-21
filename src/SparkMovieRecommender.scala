import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{ SparkConf, SparkContext }  
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

object SparkMovieRecommender {  
  
  def main(args: Array[String]) {  
      val conf = new SparkConf().setAppName("SparkMovieRecommender")  
      conf.setMaster("local")
      val sc = new SparkContext(conf)   
 
      // set log level
      Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
      Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
      
      // load data
      val data = sc.textFile("test.data")
      
       /** 
       * Product ratings are on a scale of 1-5: 
       * 5: Must see 
       * 4: Will enjoy 
       * 3: It's okay 
       * 2: Fairly bad 
       * 1: Awful 
       */  
      val ratings = data.map(_.split(',') match { case Array(user, product, rate) =>
          Rating(user.toInt, product.toInt, rate.toDouble)
        })
      
      // use ALS to set up model
      val rank = 10
      val numIterations = 20
      val model = ALS.train(ratings, rank, numIterations, 0.01)
      
      // get user and product from rating
      val usersProducts = ratings.map { case Rating(user, product, rate) =>
        (user, product)
      }
      
       // make prediction 
      val predictions = 
        model.predict(usersProducts).map { case Rating(user, product, rate) => 
          ((user, product), rate)
        }
      
       // merge actual and prediction
      val ratesAndPreds = ratings.map { case Rating(user, product, rate) => 
        ((user, product), rate)
      }.join(predictions).sortByKey()  //ascending or descending 
      
      // compute MSE
      val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) => 
        val err = (r1 - r2)
        err * err
      }.mean()
      
      // print MSE
      println("Mean Squared Error = " + MSE)  
      //Mean Squared Error = 1.37797097094789E-5
  }
}