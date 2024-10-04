import os
import json
import pandas as pd
import dill
import logging

def get_latest_model_path(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError("No model files found in the directory.")
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

def predict():
    # Установите путь к директории с JSON файлами и путь для сохранения модели
    project_path = os.environ.get('PROJECT_PATH', '..')
    test_data_path = os.path.join(project_path, 'data/test')
    model_dir = os.path.join(project_path, 'data/models')
    output_path = os.path.join(project_path, 'data/predictions/predictions.csv')

    # Найдем все JSON файлы в директории
    json_files = [f for f in os.listdir(test_data_path) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("No JSON files found in the directory.")

    # Найти последнюю модель
    model_path = get_latest_model_path(model_dir)

    # Загружаем модель
    with open(model_path, 'rb') as file:
        model = dill.load(file)

    # Считываем данные из всех JSON файлов и объединяем их в один DataFrame
    data_frames = []
    for json_file in json_files:
        with open(os.path.join(test_data_path, json_file), 'r') as file:
            data = json.load(file)
            if isinstance(data, dict):
                data = [data]  # Преобразуем словарь в список словарей
            data_frames.append(pd.DataFrame(data))

    data = pd.concat(data_frames, ignore_index=True)

    # Делаем предсказания
    predictions = model.predict(data)

    # Сохраняем только id и предсказания в новый DataFrame
    output_df = pd.DataFrame({
        'id': data['id'],
        'prediction': predictions
    })

    # Сохраняем предсказания в CSV файл
    output_df.to_csv(output_path, index=False)
    logging.info(f'Predictions are saved to {output_path}')

if __name__ == '__main__':
    predict()