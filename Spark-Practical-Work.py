# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType
from pyspark.sql.functions import expr, when, col, concat_ws, format_string
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
import sys




def main():
    
    '''
    1.-The Data
    
    For this exercise, students will use data published by the US Department of Transportation. This
    data can be downloaded from the following URL:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7
    The dataset is divided into several independent files, to make download easier. You do not need
    to download and use the entire dataset. A small piece should be sufficient, one that fits in your
    development environment and does not take too long to process. The Spark application you
    develop, however, should be able to work with any subset of this dataset, and not be limited to a
    specific piece.
    '''
    
    
    def create_df(number = 10):
        def get_csv_path(csv_path = ["dataverse_files/year-csv/1987.csv"]):
            # Example of input arguments
            # (["dataverse_files/year-csv/1987.csv","dataverse_files/year-csv/1988.csv","dataverse_files/year-csv/1989.csv" \
            #,"dataverse_files/year-csv/1990.csv","dataverse_files/year-csv/1991.csv","dataverse_files/year-csv/1992.csv" \
            #,"dataverse_files/year-csv/1993.csv","dataverse_files/year-csv/1994.csv","dataverse_files/year-csv/1995.csv" \
            #,"dataverse_files/year-csv/1996.csv","dataverse_files/year-csv/1997.csv","dataverse_files/year-csv/1998.csv" \
            #,"dataverse_files/year-csv/1999.csv","dataverse_files/year-csv/2000.csv","dataverse_files/year-csv/2001.csv" \
            #,"dataverse_files/year-csv/2002.csv","dataverse_files/year-csv/2003.csv","dataverse_files/year-csv/2004.csv" \
            #,"dataverse_files/year-csv/2005.csv","dataverse_files/year-csv/2006.csv","dataverse_files/year-csv/2007.csv" \
            #,"dataverse_files/year-csv/2008.csv])"
            csv_path1 = sys.argv[1:]
            print(csv_path1)
            if(len(csv_path1) == 0):
                print("No CSV file imput arguments (arg1,arg2,...,argn)")
                print("Using default CSV file: dataverse_files/year-csv/1987.csv")
                return csv_path
            else:
                return csv_path1
        
        def load_csv(spark, schema, csv_path):
            df = spark.read.options(header=True, nanValue="NA", emptyValue="") \
                .schema(schema) \
                .csv(csv_path)
            return df
        
        def get_schema():
            schema = StructType() \
                .add("Year", IntegerType(), True) \
                .add("Month", IntegerType(), True) \
                .add("DayofMonth", IntegerType(), True) \
                .add("DayOfWeek", IntegerType(), True) \
                .add("DepTime", IntegerType(), True) \
                .add("CRSDepTime", IntegerType(), True) \
                .add("ArrTime", IntegerType(), True) \
                .add("CRSArrTime", IntegerType(), True) \
                .add("UniqueCarrier", StringType(), True) \
                .add("FlightNum", IntegerType(), True) \
                .add("TailNum", IntegerType(), True) \
                .add("ActualElapsedTime", IntegerType(), True) \
                .add("CRSElapsedTime", IntegerType(), True) \
                .add("AirTime", IntegerType(), True) \
                .add("ArrDelay", IntegerType(), True) \
                .add("DepDelay", IntegerType(), True) \
                .add("Origin", StringType(), True) \
                .add("Dest", StringType(), True) \
                .add("Distance", IntegerType(), True) \
                .add("TaxiIn", IntegerType(), True) \
                .add("TaxiOut", IntegerType(), True) \
                .add("Cancelled", IntegerType(), True) \
                .add("CancellationCode", StringType(), True) \
                .add("Diverted", IntegerType(), True) \
                .add("CarrierDelay", IntegerType(), True) \
                .add("WeatherDelay", IntegerType(), True) \
                .add("NASDelay", IntegerType(), True) \
                .add("SecurityDelay", IntegerType(), True) \
                .add("LateAircraftDelay", IntegerType(), True)
            return schema
        
        def get_spark_session():
            spark = SparkSession.builder.appName("Spark Practical Work").getOrCreate()
            return spark

        session = get_spark_session()
        df = load_csv(spark = session, schema = get_schema(), csv_path = get_csv_path())

        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dataframe>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        df.show(number, truncate=False)
        return df, session
    
    df, session = create_df(10)
    
    
    '''
    2.-Forbidden variables
     The dataset consists of a single table with 29 columns. Some of these columns must not be
     used, and therefore need to be filtered at the beginning of the analysis. These are:
       
        - ArrTime
        - ActualElapsedTime
        - AirTime
        - TaxiIn
        - Diverted
        - CarrierDelay
        - WeatherDelay
        - NASDelay
        - SecurityDelay
        - LateAircraftDelay
     
     These variables contain information that is unknown at the time the plane takes off and,
     therefore, cannot be used in the prediction model.
     
     Also:
     - There is applied a filter to remove the canceled flights, because for the ArrDelay we need to know information.
     - There is applied a filter to convert the information to understandable information:
            - DayOfWeek: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
            - Date: Year/Month/Day
            - Time: Morning, Afternoon, Evening, Night
      Finally:
      - The schema of the dataframe is updated.
      
     '''
     
     
    def initial_preprocessing(df, session, number = 10):
        def remove_forbidden_variables(df, number = 10):
            df = df.drop("ArrTime") \
                .drop("ActualElapsedTime") \
                .drop("AirTime").drop("TaxiIn") \
                .drop("Diverted").drop("CarrierDelay") \
                .drop("WeatherDelay") \
                .drop("NASDelay") \
                .drop("SecurityDelay") \
                .drop("LateAircraftDelay")
            return df
    
        def filter_canceled(df):
            #The objetive of this function is to filter the canceled flights, because for the ArrDelay we need to know information
            #about the flights that arrived to the destination
            df = df.filter(expr("Cancelled == 0")) \
            .drop("CancellationCode", "Cancelled") \
            .filter(expr("CRSElapsedTime > 0")) \
            .distinct()
            return df
        
        def convert_to_understandable_information(df, session = session):
            def convert_to_week_days(df):
                df = df.withColumn("DayOfWeek",
                   when(col("DayOfWeek") == 1, "Monday")
                   .when(col("DayOfWeek") == 2, "Tuesday")
                   .when(col("DayOfWeek") == 3, "Wednesday")
                   .when(col("DayOfWeek") == 4, "Thursday")
                   .when(col("DayOfWeek") == 5, "Friday")
                   .when(col("DayOfWeek") == 6, "Saturday")
                   .when(col("DayOfWeek") == 7, "Sunday"))
                return df

            def update_schema(df,session = session):
                '''Raúl: Update the schema applying the changes'''
                original_columns = df.schema.fields

                # Create a new schema
                new_schema = StructType()

                for field in original_columns:
                    
                    if field.name in ["Date","DayOfWeek","DepTime", "CRSDepTime", "CRSArrTime"]:
                      new_schema.add(field.name, StringType())

                    else:
                    # For all other columns, keep the original data type
                      new_schema.add(field)

                # Apply the new schema to the DataFrame
                df = session.createDataFrame(df.rdd, new_schema)
                return df

            def convert_to_date(df):
                df = df.withColumn("Date",concat_ws("/",col("Year"),col("Month"),col("DayofMonth"))) \
                .drop("Year","Monthh","DayofMont") 
                return df
            
            def convert_to_time(df, list_colums_time = ["DepTime", "CRSDepTime", "CRSArrTime"]):
                for colum in list_colums_time:
                    df = df.withColumn("Temporal", format_string("%02d:%02d", (col(colum) / 100).cast("int"), (col(colum) % 100).cast("int")))
                    df = df.withColumn(colum, when((col("Temporal") > "06:00") & (col("Temporal") < "12:00") & (col("Temporal") >= "05:00") , "Morning")
                                .when((col("Temporal") > "12:00") & (col("Temporal") < "18:00"), "Afternoon")
                                .when((col("Temporal") > "17:00") & (col("Temporal") < "00:00"), "Evening")
                                .when((col("Temporal") > "17:00") & (col("Temporal") < "05:00"), "Night"))
                    df = df.drop("Temporal")
                return df
            
            
            
            df = convert_to_week_days(df)
            df = convert_to_date(df)
            df = convert_to_time(df)
            df = update_schema(df)
           
            return df
        
        def repeated_values_columns(df):
            def calculate_repeated_percentage(df):
                df = df.withColumn("match", 
                                when((col("DepTime") == col("CRSDepTime")), 1)
                                .otherwise(0))
                match_count = df.agg({"match": "sum"}).collect()[0][0]
                total_count = df.count()
                percentage = (match_count / total_count) * 100
                print("Percentage of repeated values in columns :", percentage, "%")
                df = df.drop("match")
                return percentage
            
            def remove_variables(df, column = ["DepTime","CRSDepTime"]):
                df = df.drop(*column)
                return df
            
            if calculate_repeated_percentage(df)>70:
                df = remove_variables(df, ["DepTime"])
            return df
        
        df = remove_forbidden_variables(df)
        df = filter_canceled(df)
        df = convert_to_understandable_information(df)
        df = repeated_values_columns(df)
   
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dataframe with initial preprocessing>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        df.show(number, truncate=False)
        return df
    
    df = initial_preprocessing(df, session,  40)    
    
    
    '''
    3.-Allowed variables
    Any other variable present in the dataset, and not included in the previous list, can be used for
    the model

    4.-Target variable
    The target variable for the prediction model will be ArrDelay.
    '''
    '''
    Creating the model and making predictions:
    - Preparing the data: 
      -The new string variables are indexed, to be able to fit a lr model.
      - The null values are imputed.
      - The data is splitted in test and train splits.
  
    - As our target variable is a continuous variables, we will use a linear regression model taking as input variables
    the rest of variables.
    - The model is evaluated using RMSE
    '''


    def create_model_and_predict(df, number, target_var:str, train_percent=0.8, test_percent=0.2):
  
        def prepare_data(df, number, target_var=target_var):
          
          def split(data, train_percent=0.8, test_percent=0.2):
            if train_percent < 1 and test_percent < 1 and train_percent > 0 and test_percent > 0:
                train_data, test_data = data.randomSplit([train_percent, test_percent])
            else:
                print("Invalid requested split percentages")

            return train_data, test_data
          
          def index_data(df, string_columns = ["Date","DayOfWeek", "CRSArrTime","CRSDepTime","UniqueCarrier","Origin", "Dest"]):
              for column in string_columns:
                indexer = StringIndexer(inputCol=column, outputCol=column+"_indexed", handleInvalid="skip")
                df = indexer.fit(df).transform(df)
                df = df.drop(column)
                print("<<INDEXED DATAFRAME>>")
                df.show(10)
              return df
            
          def impute_data(df):
            #TaxiOut and TailNum are completetely invalid(NULL)
            #Remove the columns
            df = df.drop("TailNum")
            df = df.drop("TaxiOut")
            
            #The rest of the columns are handled by the imputer
            imputer = Imputer(
            inputCols=df.columns, 
            outputCols=["{}_imputed".format(c) for c in df.columns])
            model = imputer.fit(df)
            df = model.transform(df)
              
            # Drop the original columns
            for col in df.columns:
              if not col.endswith("_imputed"):
                df = df.drop(col)

            return df

          def finalize_data(df,target_var = target_var):
            input_cols = [column for column in df.columns if column != target_var]

            assembler = VectorAssembler(inputCols=input_cols, outputCol="features", handleInvalid = "keep")
            output = assembler.transform(df)
            finalized_data = output.select("features", target_var)
            return finalized_data

          print("<<PREPARING THE DATA>>")
          print("<<<STEP 0: TRAIN TEST SPLIT>>>")
          data_train, data_test = split(df)
          print("<<<STEP 1: INDEX STRING VARIABLES>>")
          data_train = index_data(data_train)
          data_test = index_data(data_test)
          print("<<<STEP 2: IMPUTE NULL VALUES>>>")
          data_train = impute_data(data_train)
          data_test = impute_data(data_test)
          print("<<STEP 3: FINALIZE AND RETURN>>")
          data_train = finalize_data(data_train)
          data_test = finalize_data(data_test)
        
          return data_train, data_test

        def try_model(target_var, train_data, test_data):

          def fit(model, data):
            fitted_model = model.fit(data)
            print("<<Model has been fitted>>")
            return fitted_model
  
          def predict(model, data):
            predictions = model.transform(data)
            return predictions
        
          lr = LinearRegression(featuresCol='features', labelCol=target_var)
          fitted_lr = fit(lr, train_data)
          predictions = predict(fitted_lr, test_data)
          print("<<<Model has made predictions>>>")
          return predictions, fitted_lr

        def evaluate_model(predicted_values,target_var=target_var):
          evaluator = RegressionEvaluator(labelCol=target_var, predictionCol="prediction", metricName="rmse")
          rmse = evaluator.evaluate(predicted_values)

          print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
          return rmse

        
        
        data_train, data_test = prepare_data(df, number=number, target_var=target_var)
        predictions, fitted_model = try_model(target_var, data_train, data_test)
        rmse = evaluate_model(predicted_values = predictions)

        return predictions, fitted_model, rmse
    
    
    '''
    Default target variable is ArrDelay. As data has already been processed,
    the rest of the columns serve as input.
    String columns are fed to the model indexed to avoid errors.
    '''

    '''predicted_arr_delays, model, rmse =''' 
    create_model_and_predict(df, number = 10, target_var = "ArrDelay_imputed")

    session.stop()

if __name__ == "__main__":
    main()