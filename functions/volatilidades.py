import polars as pl


def volatilidades(df: pl.DataFrame):

    df = df.with_columns((abs(pl.col('close') - pl.col('open')) / pl.col('open')).alias('volatilidad'))
    volatilidad_mean = df['volatilidad'].mean()
    volatilidad_std = df['volatilidad'].std()

    maximo_volat_alta = volatilidad_mean + (2 * volatilidad_std)
    minimo_volat_alta = volatilidad_mean + volatilidad_std

    df = df.with_columns([
        pl.when(pl.col('volatilidad') >= maximo_volat_alta)
          .then(pl.lit('volat-outlayer'))
          .when(pl.col('volatilidad').is_between(minimo_volat_alta, maximo_volat_alta, closed='none'))
          .then(pl.lit('volat-alta'))
          .when(pl.col('volatilidad').is_between(volatilidad_mean, minimo_volat_alta, closed='both'))
          .then(pl.lit('volat-normal-alta'))
          .when(pl.col('volatilidad') < volatilidad_mean)
          .then(pl.lit('volat-normal'))
          .otherwise(pl.lit('otro')).alias("tipo_volatilidad")
    ])
    return df
