# DetecShips_Atech_MultpleModels
Detecção de objetos para empresa ATECH, com modelos Mobilev2 ssd fpn lite 320x320, Yolov5n e Fomo, destacando um relatório sobre o desempenho destes.

**Cada diretório contem o notebook e os resultados obtidos**

- Fomo no caso é de propriedade do edge impulse então o notebook contém informações sobre resutados obtidos e um txt contem o link do projeto do FOMO
- MobileNet SSD FPN LITE 320x320 contém o Notebook que é basicamente o passo a passo inteiro para executar tudo que foi feito, comentado, em inglês, algumas alterações foram necessárias serem feitas em códigos de outras pessoas para que funcione corretamente todo notebook, isto também está comentado
- Yolov5n também contém todo passo a passo para execução do modelo com os resultados e como foi obtido.
- EfficientDet0 também contém todo passo a passo para execução do modelo com resultados obtidos e o modelo tflite

    
      Observação: Até o momento o melhor modelo é o Yolov5n float32 ou float16, que obteve o melhor resultado
  
- A base de dados vem do roboflow, basta exportar como jupyter que é gerado um link e assim conseguimos ter a base de dados no notebook colaboratory.


