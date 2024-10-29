from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, regexp_extract, col, sum as spark_sum, avg as spark_avg, min as spark_min, max as spark_max

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("ComplexSerialKillersAnalysis").getOrCreate()

# Leer el archivo CSV desde HDFS
df = spark.read.csv("hdfs://localhost:9000/Tarea3/serial_killers.csv", header=True, inferSchema=True)

# Separar la columna "Country" en una lista de países
df = df.withColumn("Country", split(df["Country"], ", "))

# Explode la columna "Country" para crear una fila separada para cada país
df = df.withColumn("Country", explode(df["Country"]))

# Extraer los números de "Proven victims" y "Possible victims"
df = df.withColumn("Proven_victims", regexp_extract(df["Proven victims"], r'\d+', 0).cast("int"))
df = df.withColumn("Possible_victims", regexp_extract(df["Possible victims"], r'\d+', 0).cast("int"))

# Reemplazar valores nulos con 0 para las columnas de víctimas
df = df.fillna({"Proven_victims": 0, "Possible_victims": 0})

# Crear una columna de "Total_victims" que suma "Proven_victims" y "Possible_victims"
df = df.withColumn("Total_victims", col("Proven_victims") + col("Possible_victims"))

# Crear otra columna para el rango de años
df = df.withColumn("Years_active_start", regexp_extract(df["Years active"], r'(\d{4})', 0).cast("int"))
df = df.withColumn("Years_active_end", regexp_extract(df["Years active"], r'(\d{4})$', 0).cast("int"))

# Filtrar y agrupar los datos en múltiples pasos para un pipeline más largo

# Paso 1: Agrupar por país y obtener estadísticas de víctimas
country_victims = df.groupBy("Country").agg(
    spark_sum("Proven_victims").alias("Total_Proven_victims"),
    spark_sum("Possible_victims").alias("Total_Possible_victims"),
    spark_sum("Total_victims").alias("Total_victims"),
    spark_avg("Proven_victims").alias("Average_Proven_victims"),
    spark_min("Years_active_start").alias("Earliest_year"),
    spark_max("Years_active_end").alias("Latest_year")
)

# Paso 2: Calcular el rango de años activos para cada país
country_victims = country_victims.withColumn("Years_active_range", col("Latest_year") - col("Earliest_year"))

# Paso 3: Filtrar países con más de 50 víctimas confirmadas
country_victims = country_victims.filter(col("Total_Proven_victims") > 50)

# Paso 4: Calcular un promedio ajustado de víctimas basado en el rango de años
country_victims = country_victims.withColumn("Adjusted_Avg_Victims_Per_Year", col("Total_victims") / col("Years_active_range"))

# Ordenar por número total de víctimas en orden descendente y mostrar
country_victims = country_victims.orderBy(col("Total_victims").desc())

country_victims.show(10)
