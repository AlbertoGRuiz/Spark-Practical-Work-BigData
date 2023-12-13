# Import libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType
from pyspark.sql.functions import when, lit, col, length, concat_ws, substring, avg, desc, date_format, expr, format_string
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import SparseVector
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, Normalizer, UnivariateFeatureSelector
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
    
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
            return csv_path
        
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

        
        df = load_csv(spark = get_spark_session(), schema = get_schema(), csv_path = get_csv_path())
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dataframe>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        df.show(number, truncate=False)
        return df
    
    df = create_df(10)
    
    
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
     - There is applied a filter to convert the information to understable information:
            - DayOfWeek: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
            - Date: Year/Month/Day
            - Time: Morning, Afternoon, Evening, Night
     '''

    def initial_preprocessing(df, number = 10):
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
        
        def convert_to_undestable_information(df):
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
            
            def convert_to_date(df):
                df = df.withColumn("Date",concat_ws("/",col("Year"),col("Month"),col("DayofMonth"))) \
                .drop("Year","Monthh","DayofMont") 
                return df
            
            def convert_to_time(df, list_colums_time = ["DepTime", "CRSDepTime", "CRSArrTime"]):
                df = df.withColumn("Time", format_string("%02d:%02d", (col("DepTime") / 100).cast("int"), (col("DepTime") % 100).cast("int")))
                for colum in list_colums_time:
                    df = df.withColumn(colum, when((col("Time") > "06:00") & (col("Time") < "12:00") & (col("Time") >= "05:00") , "Morning")
                                .when((col("Time") > "12:00") & (col("Time") < "18:00"), "Afternoon")
                                .when((col("Time") > "17:00") & (col("Time") < "00:00"), "Evening")
                                .when((col("Time") > "17:00") & (col("Time") < "05:00"), "Night"))
                df.drop("Time")
                return df
            
            df = convert_to_week_days(df)
            df = convert_to_date(df)
            df = convert_to_time(df)
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
        df = convert_to_undestable_information(df)
        df = repeated_values_columns(df)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dataframe with initial preprocessing>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        df.show(number, truncate=False)
        return df
    
    df = initial_preprocessing(df, 40)
    
    '''
    3.-Allowed variables
    Any other variable present in the dataset, and not included in the previous list, can be used for
    the model

    4.-Target variable
    The target variable for the prediction model will be ArrDelay.
    '''
     

    




    # 4. Dividir el conjunto de datos en entrenamiento y prueba
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

    def df_remove_null(df):
        df = remove_null_rows_target(df)
        df = remove_null_columns_and_impute(df)
        return df
    def remove_null_rows_target(df):
        df = df.na.drop(subset=['ArrDelay']) # Drop rows with null values in ArrDelay column
        return df
    def remove_null_columns_and_impute(df):
        total = df.count()
        print("Total rows:", total)
        for colum in df.columns:
            not_null = df.filter(col(colum).isNotNull()).count() / total * 100
            null_percentage = 100-not_null
            print(colum, "null:", null_percentage, '%')
            
            if null_percentage > 50: # If more than 50% of the values are null, drop the column
                print("-------------------------->> Drop column", colum)
                df = df.drop(colum)
            
            elif null_percentage <= 50 and null_percentage > 0: # If less than 50% of the values are null (but exists nu), impute the column
                print("-------------------------->> Imput column", colum)
                imputer = Imputer(inputCols=[colum], outputCols=[colum])
                imputer_model = imputer.fit(df)
                df = imputer_model.transform(df)
                print("NEW", colum, "null:", 100-(df.filter(col(colum).isNotNull()).count() / total * 100), '%')
        return df

    train_data = df_remove_null(train_data)
    test_data = df_remove_null(test_data)
    
    
    def get_string_columns(df):
        return [name for name, dtype in df.dtypes if dtype == 'string']
    def categorical_to_numeric(df, categorical_columns):
        for column in categorical_columns:
            string_indexer = StringIndexer(inputCol=column, outputCol=column+"Index")
            df = string_indexer.fit(df).transform(df)
            df = df.drop(column)
        return df


    train_data = categorical_to_numeric(train_data, get_string_columns(train_data))
    test_data = categorical_to_numeric(test_data, get_string_columns(test_data))
    train_data.show(1024, truncate=False)


    def vector_assembler(df):
        feature_columns = df.drop("ArrDelay").columns
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = assembler.transform(df)
        return df, feature_columns


    train_data, feature_columns = vector_assembler(train_data)
    test_data, feature_columns = vector_assembler(test_data)

    # 5. Aplicar la regresión lineal
    lr = LinearRegression(featuresCol="features", labelCol="ArrDelay")
    model = lr.fit(train_data)

    # Imprimir coeficientes e intercepto
    print("Coefficients: {}".format(model.coefficients))
    print("Intercept: {}".format(model.intercept))

    # Realizar predicciones en el conjunto de prueba
    predictions = model.transform(test_data)

    # Mostrar algunas predicciones
    predictions.select("ArrDelay", "prediction", *feature_columns).show(10)

    df_features = df.drop("ArrDelay")
    columns_feature = df_features.columns
    columns_target = ['ArrDelay']




    
if __name__ == "__main__":
    main()

