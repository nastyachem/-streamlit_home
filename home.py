
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Настройка страницы
st.set_page_config(
    page_title="Сервис предсказания стоимости недвижимости",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Словарь для переименования колонок
RENAME_DICT = {
    'Id': 'id',
    'MSSubClass': 'ms_sub_class',
    'MSZoning': 'ms_zoning',
    'LotFrontage': 'lot_frontage',
    'LotArea': 'lot_area',
    'Street': 'street',
    'Alley': 'alley',
    'LotShape': 'lot_shape',
    'LandContour': 'land_contour',
    'Utilities': 'utilities',
    'LotConfig': 'lot_config',
    'LandSlope': 'land_slope',
    'Neighborhood': 'neighborhood',
    'Condition1': 'condition1',
    'Condition2': 'condition2',
    'BldgType': 'bldg_type',
    'HouseStyle': 'house_style',
    'OverallQual': 'overall_qual',
    'OverallCond': 'overall_cond',
    'YearBuilt': 'year_built',
    'YearRemodAdd': 'year_remod_add',
    'RoofStyle': 'roof_style',
    'RoofMatl': 'roof_matl',
    'Exterior1st': 'exterior1st',
    'Exterior2nd': 'exterior2nd',
    'MasVnrType': 'mas_vnr_type',
    'MasVnrArea': 'mas_vnr_area',
    'ExterQual': 'exter_qual',
    'ExterCond': 'exter_cond',
    'Foundation': 'foundation',
    'BsmtQual': 'bsmt_qual',
    'BsmtCond': 'bsmt_cond',
    'BsmtExposure': 'bsmt_exposure',
    'BsmtFinType1': 'bsmt_fin_type1',
    'BsmtFinSF1': 'bsmt_fin_sf1',
    'BsmtFinType2': 'bsmt_fin_type2',
    'BsmtFinSF2': 'bsmt_fin_sf2',
    'BsmtUnfSF': 'bsmt_unf_sf',
    'TotalBsmtSF': 'total_bsmt_sf',
    'Heating': 'heating',
    'HeatingQC': 'heating_qc',
    'CentralAir': 'central_air',
    'Electrical': 'electrical',
    '1stFlrSF': '1st_flr_sf',
    '2ndFlrSF': '2nd_flr_sf',
    'LowQualFinSF': 'low_qual_fin_sf',
    'GrLivArea': 'gr_liv_area',
    'BsmtFullBath': 'bsmt_full_bath',
    'BsmtHalfBath': 'bsmt_half_bath',
    'FullBath': 'full_bath',
    'HalfBath': 'half_bath',
    'BedroomAbvGr': 'bedroom_abv_gr',
    'KitchenAbvGr': 'kitchen_abv_gr',
    'KitchenQual': 'kitchen_qual',
    'TotRmsAbvGrd': 'tot_rms_abv_grd',
    'Functional': 'functional',
    'Fireplaces': 'fireplaces',
    'FireplaceQu': 'fireplace_qu',
    'GarageType': 'garage_type',
    'GarageYrBlt': 'garage_yr_blt',
    'GarageFinish': 'garage_finish',
    'GarageCars': 'garage_cars',
    'GarageArea': 'garage_area',
    'GarageQual': 'garage_qual',
    'GarageCond': 'garage_cond',
    'PavedDrive': 'paved_drive',
    'WoodDeckSF': 'wood_deck_sf',
    'OpenPorchSF': 'open_porch_sf',
    'EnclosedPorch': 'enclosed_porch',
    '3SsnPorch': '3ssn_porch',
    'ScreenPorch': 'screen_porch',
    'PoolArea': 'pool_area',
    'PoolQC': 'pool_qc',
    'Fence': 'fence',
    'MiscFeature': 'misc_feature',
    'MiscVal': 'misc_val',
    'MoSold': 'mo_sold',
    'YrSold': 'yr_sold',
    'SaleType': 'sale_type',
    'SaleCondition': 'sale_condition',
    'SalePrice': 'sale_price'
}

# Функция для подготовки данных
def prepare_data(df):
    # Копируем датафрейм
    df_prep = df.copy()
    
    # Переименование колонок согласно словарю
    columns_to_rename = {}
    for orig_col, new_col in RENAME_DICT.items():
        if orig_col in df_prep.columns:
            columns_to_rename[orig_col] = new_col
    
    # Переименуем колонки
    if columns_to_rename:
        df_prep.rename(columns=columns_to_rename, inplace=True)
        st.info(f"Переименовано {len(columns_to_rename)} колонок из CamelCase в snake_case")
    
    # Создаём целевую переменную логарифмированную, если нужно
    if 'sale_price' in df_prep.columns and 'sale_price_log' not in df_prep.columns:
        df_prep['sale_price_log'] = np.log1p(df_prep['sale_price'])
    
    # Информация о пропущенных значениях
    missing_values = df_prep.isnull().sum()
    if missing_values.any():
        missing_cols = missing_values[missing_values > 0]
        st.warning(f"Обнаружено {missing_cols.sum()} пропущенных значений в {len(missing_cols)} колонках. Они будут обработаны автоматически.")
        
    return df_prep

# Функция для обучения модели
def train_model(df):
    try:
        # Проверяем наличие целевой переменной
        if 'sale_price_log' not in df.columns:
            st.error("В тренировочном файле отсутствует целевая переменная 'sale_price_log'")
            return None, None
        
        # Подготовка данных
        features = df.drop(['id', 'sale_price', 'sale_price_log'], axis=1, errors='ignore')
        target = df['sale_price_log']
        
        # Определяем типы признаков
        numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = features.select_dtypes(include=['object']).columns.tolist()
        
        # Создаем трансформеры для числовых и категориальных признаков с обработкой пропущенных значений
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Заполняем пропуски медианой
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Заполняем пропуски наиболее частым значением
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Создаем препроцессор с обработкой пропущенных значений
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Создаем пайплайн модели
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Разделение на тренировочные и тестовые данные
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return model, (r2, rmse)
        
    except Exception as e:
        st.error(f"Ошибка при обучении модели: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None, None

# Функция для предсказания
def predict(model, df):
    try:
        # Проверяем, что у нас есть модель
        if model is None:
            st.error("Модель не обучена")
            return None
        
        # Предсказание
        predictions_log = model.predict(df)
        predictions = np.expm1(predictions_log)
        
        return predictions, predictions_log
    except Exception as e:
        st.error(f"Ошибка при предсказании: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None, None

# Функция для отображения результатов
def display_results(predictions, df):
    # Создаем DataFrame с результатами
    results = df.copy()
    results['Предсказанная цена'] = predictions
    
    # Отображаем результаты
    st.subheader("Результаты предсказания")
    st.dataframe(results)
    
    # Визуализация
    # st.subheader("Визуализация")
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     # Гистограмма предсказанных цен
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     sns.histplot(results['Предсказанная цена'], bins=20, kde=True)
    #     plt.title('Распределение предсказанных цен')
    #     plt.xlabel('Цена')
    #     plt.ylabel('Количество')
    #     st.pyplot(fig)
    
    # with col2:
    #     # Боксплот предсказанных цен
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     sns.boxplot(y=results['Предсказанная цена'])
    #     plt.title('Боксплот предсказанных цен')
    #     plt.ylabel('Цена')
    #     st.pyplot(fig)
    
    # # Топ-5 самых дорогих домов
    # st.subheader("Топ-5 домов с наибольшей предсказанной ценой")
    # top5 = results.sort_values(by='Предсказанная цена', ascending=False).head(5)
    # st.dataframe(top5)
    
    # Кнопка для скачивания результатов
    csv = results.to_csv(index=False)
    st.download_button(
        label="Скачать результаты",
        data=csv,
        file_name='real_estate_predictions.csv',
        mime='text/csv'
    )

# Основная функция приложения
def main():
    # Боковая панель
    st.sidebar.title("Информация")
    st.sidebar.info("""
    
    Для работы загрузите сначала тренировочный датасет, затем файл для предсказаний.
   
    """)
    
    # Загрузка тренировочного файла
    st.sidebar.subheader("Загрузите тренировочный файл")
    train_file = st.sidebar.file_uploader("Тренировочный CSV", type=['csv'], key="train")
    
    # Переменные для хранения модели и метрик
    model = None
    metrics = None
    
    # Если загружен тренировочный файл
    if train_file:
        # Загружаем данные
        df_train = pd.read_csv(train_file)
        
        # Подготавливаем данные
        df_train = prepare_data(df_train)
        
        # Отображаем информацию о данных
        st.sidebar.write(f"Загружено {df_train.shape[0]} строк и {df_train.shape[1]} столбцов")
        st.sidebar.write("Пример данных:")
        st.sidebar.dataframe(df_train.head(3))
        
        # Обучаем модель
        st.sidebar.text("Обучение модели...")
        model, metrics = train_model(df_train)
        
        # Отображаем метрики модели
        if model and metrics:
            r2, rmse = metrics
            st.sidebar.success(f"Модель успешно обучена!")
            st.sidebar.metric("R² (коэффициент детерминации)", f"{r2:.4f}")
            st.sidebar.metric("RMSE (корень из среднеквадратичной ошибки)", f"{rmse:.4f}")
    
    # Основная часть
    st.title("Сервис предсказания стоимости недвижимости")
    
    # Информация о формате данных
    # with st.expander("Информация о формате данных"):
    #     st.write("""
    #     ### Формат входных данных
        
    #     Приложение поддерживает файлы CSV с колонками в форматах CamelCase (как в API Kaggle) или snake_case.
        
    #     Примеры колонок:
    #     - `SalePrice` или `sale_price` - цена продажи (целевая переменная)
    #     - `LotArea` или `lot_area` - площадь участка
    #     - `YearBuilt` или `year_built` - год постройки
        
    #     Приложение автоматически преобразует имена колонок из CamelCase в snake_case.
    #     """)
    
    # Загрузка файла для предсказаний
    st.subheader("Загрузите файл для предсказаний")
    test_file = st.file_uploader("Тестовый CSV", type=['csv'], key="test")
    
    # Если загружен файл для предсказаний и модель обучена
    if test_file and model:
        # Загружаем данные
        df_test = pd.read_csv(test_file)
        
        # Подготавливаем данные
        df_test = prepare_data(df_test)
        
        # Отображаем информацию о данных
        st.write(f"Загружено {df_test.shape[0]} строк и {df_test.shape[1]} столбцов")
        st.write("Предпросмотр данных:")
        st.dataframe(df_test.head())
        
        # Кнопка для запуска предсказания
        if st.button("Выполнить предсказание"):
            st.text("Выполнение предсказаний...")
            # Удаляем лишние колонки, если они есть
            df_test_features = df_test.drop(['id', 'sale_price', 'sale_price_log'], axis=1, errors='ignore')
            
            # Получаем предсказания
            predictions, _ = predict(model, df_test_features)
            
            # Отображаем результаты
            if predictions is not None:
                display_results(predictions, df_test)
    
    # Если модель не обучена
    elif test_file and not model:
        st.warning("Пожалуйста, сначала загрузите тренировочный файл и обучите модель.")
    
    # Если файл для предсказаний не загружен
    elif not test_file and model:
        st.info("Пожалуйста, загрузите файл с данными для получения предсказаний.")

# Запуск приложения
if __name__ == "__main__":
    main()