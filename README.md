Это решение конкурсной задачи по подстету и определению весов гранул проппанта
rosneft-count.ipynb - подсчет гранул
rosneft-dist.ipynb - вычисление распределения вечов

Для вычисление распределения вечов я настроил нейросеть mobilentv2 без аугментаций, так как организаторы сами уже аугметнировали данные

Для подсчета использовал mobilenet с замороженными слоями енкодера, так как данных для обучения подсчету было очень мало. 
