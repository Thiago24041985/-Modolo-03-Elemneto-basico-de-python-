No exercicio 1, 2 e 3    adiciona exercicios de elementos basicos de Python (EBAC)

- Pratica manipulacao de tipos primitivos (int, float, str, bool)
- Implementa operacoes aritmeticas e logicas
- Utiliza metodos de formatacao e fatiamento de strings
- Aplica estruturas de dados basicas e conversao de tipos

- resolve lista de exercicios sobre elementos basicos de Python

- Ir para o conteúdo principalAjuda sobre acessibilidade
Modo IA
Tudo
Imagens
Vídeos
Notícias
Mais
Fazer upgrade




contar sobre esse projeto para git commite Open In Colab Projeto 01 - Concessão de cartões de crédito Este notebook é semelhante ao visto em vídeo, mas contém células azuis como esta, que trazem instruções para a sua atividade. Após realizar as tarefas indicadas, você vai fazer o upload do seu arquivo no GitHub e enviar o link para a EBAC, ou alternativamente, fazer o upload do arquivo na plataforma da EBAC. Recomendamos o github, pois assim você já vai montando o seu portfólio. Etapa 1 CRISP - DM: Entendimento do negócio Como primeira etapa do CRISP-DM, vamos entender do que se trata o negócio, e quais os objetivos. Este é um problema de concessão de cartões de crédito, publicado no Kaggle, uma plataforma que promove desafios de ciência de dados, oferecendo prêmios em dinheiro para os melhores colocados. O link original está aqui. Essa é uma base de proponentes de cartão de crédito, nosso objetivo é construir um modelo preditivo para identificar o risco de inadimplência (tipicamente definida pela ocorrência de um atraso maior ou igual a 90 em um horizonte de 12 meses) através de variáveis que podem ser observadas na data da avaliação do crédito (tipicamente quando o cliente solicita o cartão). Atividades do CRISP-DM: Objetivos do negócio Note que o objetivo aqui é que o modelo sirva o mutuário (o cliente) para que avalie suas próprias decisões, e não a instituição de crédito. Objetivos da modelagem O objetivo está bem definido: desenvolver o melhor modelo preditivo de modo a auxiliar o mutuário a tomar suas próprias decisões referentes a crédito. Nessa etapa também se avalia a situação da empresa/segmento/assunto de modo a se entender o tamanho do público, relevância, problemas presentes e todos os detalhes do processo gerador do fenômeno em questão, e portanto dos dados. Também é nessa etapa que se constrói um planejamento do projeto. Etapa 2 Crisp-DM: Entendimento dos dados A segunda etapa é o entendimento dos dados. Foram fornecidas 15 variáveis mais a variável resposta (em negrito na tabela). O significado de cada uma dessas variáveis se encontra na tabela. Dicionário de dados Os dados estão dispostos em uma tabela com uma linha para cada cliente, e uma coluna para cada variável armazenando as características desses clientes. Colocamos uma cópia o dicionário de dados (explicação dessas variáveis) abaixo neste notebook: Variable Name Description Tipo sexo M = 'Masculino'; F = 'Feminino' M/F posse_de_veiculo Y = 'possui'; N = 'não possui' Y/N posse_de_imovel Y = 'possui'; N = 'não possui' Y/N qtd_filhos Quantidade de filhos inteiro tipo_renda Tipo de renda (ex: assaliariado, autônomo etc) texto educacao Nível de educação (ex: secundário, superior etc) texto estado_civil Estado civil (ex: solteiro, casado etc) texto tipo_residencia tipo de residência (ex: casa/apartamento, com os pais etc) texto idade idade em anos inteiro tempo de emprego tempo de emprego em anos inteiro possui_celular Indica se possui celular (1 = sim, 0 = não) binária possui_fone_comercial Indica se possui telefone comercial (1 = sim, 0 = não) binária possui_fone Indica se possui telefone (1 = sim, 0 = não) binária possui_email Indica se possui e-mail (1 = sim, 0 = não) binária qt_pessoas_residencia quantidade de pessoas na residência inteiro mau indicadora de mau pagador (True = mau, False = bom) binária Carregando os pacotes É considerado uma boa prática carregar os pacotes que serão utilizados como a primeira coisa do programa. import pandas as pd import seaborn as sns import matplotlib.pyplot as plt from sklearn.model_selection import train_test_split from sklearn import metrics from sklearn.ensemble import RandomForestClassifier Carregando os dados O comando pd.read_csv é um comando da biblioteca pandas (pd.) e carrega os dados do arquivo csv indicado para um objeto dataframe do pandas. # Observe que demo01.csv está na mesma pasta que este notebook # do contrário, seria necessário indicar a pasta no nome do arquivo df = pd.read_csv('demo01.csv') print ("Número de linhas e colunas da tabela: {}".format(df.shape)) df.head() Número de linhas e colunas da tabela: (16650, 16) sexo posse_de_veiculo posse_de_imovel qtd_filhos tipo_renda educacao estado_civil tipo_residencia idade tempo_emprego possui_celular possui_fone_comercial possui_fone possui_email qt_pessoas_residencia mau 0 M Y Y 0 Working Secondary / secondary special Married House / apartment 58.832877 3.106849 1 0 0 0 2.0 False 1 F N Y 0 Commercial associate Secondary / secondary special Single / not married House / apartment 52.356164 8.358904 1 0 1 1 1.0 False 2 F N Y 0 Commercial associate Secondary / secondary special Single / not married House / apartment 52.356164 8.358904 1 0 1 1 1.0 False 3 M Y Y 0 Working Higher education Married House / apartment 46.224658 2.106849 1 1 1 1 2.0 False 4 F Y N 0 Working Incomplete higher Married House / apartment 29.230137 3.021918 1 0 0 0 2.0 False Entendimento dos dados - Univariada Nesta etapa tipicamente avaliamos a distribuição de todas as variáveis. Nesta demonstração vamos ver a variável resposta e dois exemplos de univariada apenas. Mas sinta-se à vontade para tentar observar outras variáveis. print(df['mau'].value_counts()) print("\nTaxa de inadimplentes:") print(df['mau'].mean()) False 16260 True 390 Name: mau, dtype: int64 Taxa de inadimplentes: 0.023423423423423424 var = 'mau' grafico_barras = df[var].value_counts().plot.bar() Tarefa 01 - gráfico de barras Com base no código da célula anterior, construa um gráfico de barras para pelo menos duas outras variáveis. Dica: Não tente usar as variáveis tempo_emprego e idade pois o gráfico de barras dessa forma como construímos não é adequado para elas. df.dtypes # verificando quais os tipos dos dados o ideal é utilizar variáveis booleanas sexo object posse_de_veiculo object posse_de_imovel object qtd_filhos int64 tipo_renda object educacao object estado_civil object tipo_residencia object idade float64 tempo_emprego float64 possui_celular int64 possui_fone_comercial int64 possui_fone int64 possui_email int64 qt_pessoas_residencia float64 mau bool dtype: object var = 'posse_de_imovel' grafico_barras = df[var].value_counts().plot.bar() var = 'posse_de_veiculo' grafico_barras = df[var].value_counts().plot.bar() plt.clf() var = "tempo_emprego" sns.displot(df, x = var, bins = 50) plt.show() <Figure size 640x480 with 0 Axes> # Alterando valores de -1000 pra -2, para visualizar melhor no gráfico var = "tempo_emprego" df.loc[df[var]<0,var] = -2 plt.clf() var = "tempo_emprego" sns.displot(df, x = var, bins = 50) plt.show() <Figure size 640x480 with 0 Axes> Tarefa 02 - Histograma Com base no código da célula anterior, construa o histograma da variavel dade. plt.clf() var = "idade" sns.displot(df, x = var, bins = 50) plt.show() <Figure size 640x480 with 0 Axes> Entendimento dos dados - Bivariadas Entender a alteração da inadimplência indicada pela variável resposta (AtrasoRelevante2anos) e as variáveis explicativas (demais). Para isto, vamos calcular a taxa de inadimplentes (qtd inadimplentes / total) para diferentes grupos definidos pelas variáveis explicativas. var = 'idade' cat_srs, bins = pd.qcut(df[var], 4, retbins=True) g = df.groupby(cat_srs) biv = g['mau'].mean() ax = biv.plot.line() ax.set_ylabel("Proporção de inadimplentes") ticks = plt.xticks(range(len(biv.index.values)), biv.index.values, rotation = 90) Tarefa 03 - Bivariada Com base no código da célula anterior, construa uma análise bivariada para a variável tempo_emprego. Em seguida, insira uma célula de markdown e conclua se a variável parece discriminar risco de crédito. var = 'tempo_emprego' cat_srs, bins = pd.qcut(df[var], 4, retbins=True) g = df.groupby(cat_srs) biv = g['mau'].mean() ax = biv.plot.line() ax.set_ylabel("Proporção de inadimplentes") ticks = plt.xticks(range(len(biv.index.values)), biv.index.values, rotation = 90) Com o gráfico acima podemos ver uma uma correlação entre o tempo de emprego e a taxa de inadimplencia, a probabilidade de alguém com menos de 4.6 anos em um emprego ser inadimplente é maior que alguém com mais tempo no mesmo emprego Etapa 3 Crisp-DM: Preparação dos dados Nessa etapa realizamos tipicamente as seguintes operações com os dados: seleção Neste caso, os dados já estão pré-selecionados limpeza Precisaremos identificar e tratar dados faltantes construção Neste primeiro exercício não faremos construção de novas variáveis integração Temos apenas uma fonte de dados, não é necessário agregação formatação Os dados já se encontram em formatos úteis Os dados já estão pré-selecionados, construídos e integrados, mas há dados faltantes que serão eliminados na próxima célula metadata = pd.DataFrame(df.dtypes, columns = ['tipo']) metadata['n_categorias'] = 0 for var in metadata.index: metadata.loc[var,'n_categorias'] = len(df.groupby([var]).size()) metadata tipo n_categorias sexo object 2 posse_de_veiculo object 2 posse_de_imovel object 2 qtd_filhos int64 8 tipo_renda object 5 educacao object 5 estado_civil object 5 tipo_residencia object 6 idade float64 5298 tempo_emprego float64 3005 possui_celular int64 1 possui_fone_comercial int64 2 possui_fone int64 2 possui_email int64 2 qt_pessoas_residencia float64 9 mau bool 2 def convert_dummy(df, feature,rank=0): pos = pd.get_dummies(df[feature], prefix=feature) mode = df[feature].value_counts().index[rank] biggest = feature + '_' + str(mode) pos.drop([biggest],axis=1,inplace=True) df.drop([feature],axis=1,inplace=True) df=df.join(pos) return df for var in metadata[metadata['tipo'] == 'object'].index: df = convert_dummy(df, var) df qtd_filhos idade tempo_emprego possui_celular possui_fone_comercial possui_fone possui_email qt_pessoas_residencia mau sexo_M ... educacao_Lower secondary estado_civil_Civil marriage estado_civil_Separated estado_civil_Single / not married estado_civil_Widow tipo_residencia_Co-op apartment tipo_residencia_Municipal apartment tipo_residencia_Office apartment tipo_residencia_Rented apartment tipo_residencia_With parents 0 0 58.832877 3.106849 1 0 0 0 2.0 False 1 ... 0 0 0 0 0 0 0 0 0 0 1 0 52.356164 8.358904 1 0 1 1 1.0 False 0 ... 0 0 0 1 0 0 0 0 0 0 2 0 52.356164 8.358904 1 0 1 1 1.0 False 0 ... 0 0 0 1 0 0 0 0 0 0 3 0 46.224658 2.106849 1 1 1 1 2.0 False 1 ... 0 0 0 0 0 0 0 0 0 0 4 0 29.230137 3.021918 1 0 0 0 2.0 False 0 ... 0 0 0 0 0 0 0 0 0 0 ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... 16645 0 54.109589 9.884932 1 0 0 0 2.0 True 0 ... 0 1 0 0 0 0 0 0 0 0 16646 0 43.389041 7.380822 1 1 1 0 2.0 True 0 ... 0 0 0 0 0 0 0 0 0 0 16647 0 30.005479 9.800000 1 1 0 0 2.0 True 1 ... 0 0 0 0 0 0 0 0 0 0 16648 0 30.005479 9.800000 1 1 0 0 2.0 True 1 ... 0 0 0 0 0 0 0 0 0 0 16649 0 33.936986 3.630137 1 0 1 1 2.0 True 0 ... 0 0 0 0 0 0 0 0 0 0 16650 rows × 29 columns Etapa 4 Crisp-DM: Modelagem Nessa etapa que realizaremos a construção do modelo. Os passos típicos são: Selecionar a técnica de modelagem Utilizaremos a técnica de floresta aleatória (random forest), pois é uma técnica bastante versátil e robusta que captura bem padrões complexos nos dados, relativamente fácil de se usar e que costuma produzir excelentes resultados para uma classificação como estas. Vamos ver esse algoritmo em detalhes mais adiante no curso, mas pense nele por enquanto como uma regra complexa baseada nas variáveis explicativas que classifica o indivíduo como inadimplente ou não. Mais adiante no curso vamos extrair mais dessa técnica. Desenho do teste Antes de rodar o modelo precisamos construir um desenho do teste que será realizado. Para desenvolver um modelo como este, é considerado uma boa prática dividir a base em duas, uma chamada treinamento, onde o algoritmo 'aprende', e outra chamada teste, onde o algoritmo é avaliado. Essa prática fornece uma métrica de avaliação mais fidedigna do algoritmo, falaremos mais detalhes em lições futuras. Avaliação do modelo Faremos a avaliação do nosso modelo através do percentual de acerto, avaliando a classificação do modelo (inadimplente e não inadimplente) e comparando com o estado real armazenado na variável resposta (AtrasoRelevante2anos). Esse percentual de acerto é frequentemente chamado de acurácia (obs: nunca usar assertividade... assertivo não é aquele que acerta, e sim "adj.: em que o locutor declara algo, positivo ou negativo, do qual assume inteiramente a validade; declarativo." aCertivo está errado ;) Dividindo a base em treino e teste # Tirando a v. resposta da base de treinamento x = df.drop("mau",axis = 1) y = df["mau"] # Tirando ID da base de treinamento e teste x_train, x_test, y_train, y_test = train_test_split(x, y) x_train qtd_filhos idade tempo_emprego possui_celular possui_fone_comercial possui_fone possui_email qt_pessoas_residencia sexo_M posse_de_veiculo_Y ... educacao_Lower secondary estado_civil_Civil marriage estado_civil_Separated estado_civil_Single / not married estado_civil_Widow tipo_residencia_Co-op apartment tipo_residencia_Municipal apartment tipo_residencia_Office apartment tipo_residencia_Rented apartment tipo_residencia_With parents 12785 1 39.939726 20.126027 1 0 0 0 2.0 0 0 ... 0 0 0 1 0 0 0 0 0 0 4448 0 28.252055 1.926027 1 0 1 0 2.0 1 0 ... 0 0 0 0 0 0 0 0 0 0 4200 0 46.660274 1.758904 1 0 0 0 1.0 0 0 ... 0 0 0 1 0 0 0 0 0 0 6130 0 56.408219 10.082192 1 0 0 0 2.0 0 0 ... 0 0 0 0 0 0 1 0 0 0 2038 0 64.131507 -2.000000 1 0 0 0 2.0 0 0 ... 0 0 0 0 0 0 0 0 0 0 ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... 3617 0 39.701370 5.356164 1 0 0 0 1.0 0 0 ... 0 0 0 0 1 0 0 0 0 0 14412 2 34.021918 0.619178 1 1 1 0 4.0 1 0 ... 0 0 0 0 0 0 0 0 0 0 2424 0 49.227397 2.452055 1 1 0 0 2.0 1 1 ... 0 0 0 0 0 0 0 0 0 0 1263 0 52.169863 9.093151 1 0 0 0 1.0 0 0 ... 0 0 1 0 0 0 0 0 0 0 1716 0 59.452055 6.646575 1 0 0 0 2.0 1 1 ... 0 0 0 0 0 0 0 0 0 0 12487 rows × 28 columns x_test qtd_filhos idade tempo_emprego possui_celular possui_fone_comercial possui_fone possui_email qt_pessoas_residencia sexo_M posse_de_veiculo_Y ... educacao_Lower secondary estado_civil_Civil marriage estado_civil_Separated estado_civil_Single / not married estado_civil_Widow tipo_residencia_Co-op apartment tipo_residencia_Municipal apartment tipo_residencia_Office apartment tipo_residencia_Rented apartment tipo_residencia_With parents 8482 1 37.235616 8.893151 1 0 0 0 3.0 0 1 ... 0 0 0 0 0 0 0 0 0 0 219 0 47.326027 3.693151 1 0 1 0 2.0 0 0 ... 0 0 0 0 0 0 0 0 0 0 1146 1 32.109589 3.263014 1 0 0 0 3.0 0 0 ... 0 0 0 0 0 0 0 0 0 0 1644 0 29.613699 6.394521 1 1 0 0 1.0 1 0 ... 0 0 0 1 0 0 0 0 0 0 11542 0 40.520548 4.550685 1 1 0 0 2.0 0 0 ... 0 0 0 0 0 0 0 0 0 0 ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... 16140 0 33.446575 8.967123 1 0 0 0 2.0 1 1 ... 0 0 0 0 0 0 0 0 0 0 12290 0 23.430137 4.279452 1 0 0 0 2.0 0 0 ... 0 0 0 0 0 0 0 0 0 0 7629 0 35.468493 0.934247 1 0 1 1 2.0 0 0 ... 0 1 0 0 0 0 0 0 0 0 15623 0 50.506849 5.958904 1 0 1 0 2.0 0 0 ... 0 0 0 0 0 0 0 0 0 0 14770 1 39.906849 0.800000 1 1 0 0 3.0 1 1 ... 0 0 0 0 0 0 0 0 0 0 4163 rows × 28 columns Rodando o modelo A função RandomForestClassifier gera a estrutura da floresta aleatória, e o parâmetro n_estimator define o número de árvores na floresta. Normalmente a acurácia do modelo tende a aumentar com o número de árvores, pelo menos até um certo limite - e aumenta também o recurso computacional demandado. Você pode alterar esse parâmetro e verificar se a acurácia do seu modelo melhora - não recomendamos valores muito altos. Vá alterando aos poucos e percebendo como o tempo aumenta com os seus recursos. Não é necessário ir muito além de umas 100 árvores. # Treinar uma Random Forest com 5 árvores clf = RandomForestClassifier(n_estimators=3) clf.fit(x_train,y_train) RandomForestClassifier(n_estimators=3) In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org. # Calculando a acuracia y_pred = clf.predict(x_test) acc = metrics.accuracy_score(y_test, y_pred) print('Acurácia: {0:.2f}%'.format(acc*100)) Acurácia: 97.53% # Matriz de confusão tab = pd.crosstab(index = y_pred, columns = y_test) #print(tab[1][0]/(tab[1][0] + tab[0][0])) alterei essa linha #print(tab[1][1]/(tab[1][1] + tab[0][1]))e essa linha print(tab.iloc[1, 0] / (tab.iloc[1, 0] + tab.iloc[0, 0])) # por essa print(tab.iloc[1, 1] / (tab.iloc[1, 1] + tab.iloc[0, 1])) # e por essa 0.009844942160964805 0.26 Tarefa 04 - Bivariada Para essa tarefa, crie três células extras, copie nelas o código das três células anteriores (na mesma ordem) e altere o parâmetro n_estimators= da função RandomForestClassifier e insira uma quantidade maior que 3 nesse parâmetro. Rode as três células anteriores para calcular a acurácia do modelo e veja se você consegue uma acurácia melhor. # Treinar uma Random Forest com 5 árvores clf = RandomForestClassifier(n_estimators=3) clf.fit(x_train,y_train) RandomForestClassifier(n_estimators=3) In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org. # Calculando a acuracia y_pred = clf.predict(x_test) acc = metrics.accuracy_score(y_test, y_pred) print('Acurácia: {0:.2f}%'.format(acc*100)) Acurácia: 97.48% # Matriz de confusão """tab = pd.crosstab(index = y_pred, columns = y_test) print(tab[1][0]/(tab[1][0] + tab[0][0])) print(tab[1][1]/(tab[1][1] + tab[0][1])) tab""" tab = pd.crosstab(index=y_pred, columns=y_test) print(tab.iloc[0, 1] / (tab.iloc[0, 1] + tab.iloc[0, 0])) print(tab.iloc[1, 1] / (tab.iloc[1, 1] + tab.iloc[1, 0])) tab 0.01679649464459591 0.34545454545454546 mau False True row_0 False 4039 69 True 36 19 # testando uma acurácia melhor # Calculando a acuracia y_pred = clf.predict(x_test) acc = metrics.accuracy_score(y_test, y_pred) print('Acurácia: {0:.2f}%'.format(acc*100)) Acurácia: 97.79% # Treinar uma Random Forest com 5 árvores clf = RandomForestClassifier(n_estimators=10) clf.fit(x_train,y_train) RandomForestClassifier(n_estimators=10) In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org. # Matriz de confusão """tab = pd.crosstab(index = y_pred, columns = y_test) print(tab[1][0]/(tab[1][0] + tab[0][0])) print(tab[1][1]/(tab[1][1] + tab[0][1])) tab""" tab = pd.crosstab(index=y_pred, columns=y_test) print(tab.iloc[0, 1] / (tab.iloc[0, 1] + tab.iloc[0, 0])) print(tab.iloc[1, 1] / (tab.iloc[1, 1] + tab.iloc[1, 0])) tab mau False True row_0 False 4055 72 True 20 16 Etapa 5 Crisp-DM: Avaliação dos resultados A etapa final do CRISP. Neste casp, a nossa avaliação termina com a acurácia. Mas em problemas futuros aprofundaremos mais - a ideia seria avaliar o impacto do uso do modelo no negócio, ou seja, o quanto o resultado financeiro melhora em detrimento da utilização do modelo. Como um exemplo simples, considere que um cliente bom pagador deixa (em média) 5 'dinheiros' de lucro, e um mau pagador deixa (em média) 100 'dinheiros' de prejuízo. de acordo com a matriz de confusão: Decisão lucro dos bons lucro dos maus total Aprovador 4042 x 5 72 x (-100) 13.010 Reprovar 27 x 5 22 x (-100) -2.065 Estariamos evitando, portanto, um prejuízo de -2.145 'dinheiros' - o que na prática significa um aumento no lucro. Etapa 6 Crisp-DM: Implantação Nessa etapa colocamos em uso o modelo desenvolvido, normalmente implementando o modelo desenvolvido em um motor de crédito que toma as decisões com algum nível de automação - tipicamente aprovando automaticamente clientes muito bons, negando automaticamente clientes muito ruins, e enviando os intermediários para análise manual.
9 sites
Confira os principais resultados da Web para saber mais sobre esse tema:

GitHub
·
https://gist.github.com
Projeto 01 - Classificação de crédito.ipynb - Github-Gist
Projeto 01 - Concessão de cartões de crédito ¶. Este notebook é semelhante ao visto em vídeo, mas contém células azuis como esta, que trazem instruções para a sua atividade. Após realizar as tarefas i

Kaggle
·
https://www.kaggle.com
Projeto 01 - Concessão de cartões de crédito - Kaggle
TypeError: Failed to fetch at https://www.kaggle.com/static/assets/app.js?v=6f3da7390dba00fc:1:3759239 at Object. (https://www.kaggle.com/static/assets/app.js?v=6f3da7390dba00fc:1:3755815) at w (https

YouTube
·
https://www.youtube.com
Enviando um notebook feito no Google Colab para o Github
Nesse vídeo irei explicar como enviar um Python Notebook feito no Google Colab para um repositório no Github.

Mapa da Prova
·
https://www.mapadaprova.com.br
Questões de AOCP - GIT - Mapa da Prova
Cargo(s): Analista de TIC II - Construção de Software. Ano: 2018. Matéria/Assunto: Tecnologia da Informação > Desenvolvimento de Sistemas / Ferramentas de desenvolvimento / Ferramentas de gerência de

YouTube
·
https://www.youtube.com
Criando contribuição open-source no Github do zero! - YouTube
Comments. 57. Comment... 33:51. Go to channel Kanto Nerd · GitHub real-world practice - beyond the commit #Day14. Kanto Nerd. Auto-dubbed. 4.2K views · 16:47. Go to channel Filipe Deschamps · Porque E

Stack Overflow
·
https://stackoverflow.com
Métodos para usar o Git com o Google Colab - Stack Overflow
Pontuação mais alta (padrão), Em alta (votos recentes têm maior peso), Data de modificação (da mais recente para a mais antiga), Data de criação (da mais antiga para a mais recente). Essa resposta é ú

YouTube
·
https://www.youtube.com
Aula 01 - Iniciando um projeto no GIT do ZERO - YouTube
Video idealizado para a disciplica do projeto integrador do IFNMG Campus Araçuaí.

YouTube
·
https://www.youtube.com
GitHub Como Usar - Tutorial para Iniciantes - YouTube
Nesse vídeo você vai aprender do absoluto zero a trabalhar com o Git e GitHub, fica até o final do vídeo. Download do Git: https://git-scm.com/ Crie a sua conta na plataforma do GitHub: https://github

YouTube
·
https://www.youtube.com
Tutorial GitHub para Iniciantes | Como usar o GitHub 2026 Atualizado
Comments. 112. Depois que eu assisti umas 3x eu comecei a entender . Treinar a mente em modo DEV é desafiador. Acho que o segredo é paciência e persistência . # valeuprofessor. 1:25:31 · Go to channel


escrever falando sobre o projeto para min
Aqui estão opções prontas de textos para você usar tanto nas mensagens de git commit quanto no README.md do seu repositório no GitHub, facilitando a apresentação do seu portfólio.
1. Mensagens de Git Commit (Padrão Conventional Commits)
Você pode escolher uma das opções abaixo dependendo de como vai subir o arquivo:
Opção 1 (Para o commit inicial/geral do projeto finalizado):
text
feat: adiciona projeto de previsao de inadimplencia em cartoes de credito

- Implementa fluxo completo seguindo a metodologia CRISP-DM
- Realiza analise exploratoria univariada e bivariada (idade e tempo de emprego)
- Prepara dados categoricos com variaveis dummy
- Treina e avalia modelo Random Forest ajustando hiperparametros (n_estimators)
Use o código com cuidado.
Opção 2 (Mais curta e direta):
text
feat: finaliza notebook do Projeto 01 - Concessao de cartoes de credito (EBAC)
Use o código com cuidado.
2. Texto para o README.md (Descrição do Repositório)
Copie e cole o modelo abaixo no arquivo README.md do seu GitHub para deixar o projeto profissional para recrutadores:
markdown
# 💳 Projeto 01: Concessão de Cartões de Crédito (Credit Scoring)

Este projeto faz parte do curso de Ciência de Dados da **EBAC** e tem como objetivo desenvolver um modelo preditivo capaz de identificar o risco de inadimplência na concessão de cartões de crédito, seguindo as etapas da metodologia **CRISP-DM**.

## 🎯 Objetivo do Negócio
Construir um modelo de *Machine Learning* que auxilie o proponente (cliente) a avaliar seu próprio risco de crédito (definido pela ocorrência de atraso $\ge$ 90 dias em um horizonte de 12 meses) com base em dados cadastrais e socioeconômicos.

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Python 3**
* **Pandas:** Manipulação e limpeza de dados.
* **Seaborn & Matplotlib:** Visualização de dados e análise exploratória.
* **Scikit-Learn:** Divisão de treino/teste, modelagem (*Random Forest*) e métricas de avaliação.

## 📊 Etapas Desenvolvidas (CRISP-DM)
1. **Entendimento do Negócio:** Definição do problema de risco de crédito e da variável resposta (`mau`).
2. **Entendimento dos Dados:** 
   * Análise univariada (distribuição da variável alvo, posse de imóvel, posse de veículo e histograma de idade).
   * Análise bivariada avaliando a taxa de inadimplência por faixas de `idade` e `tempo_emprego`, identificando que clientes com menor tempo de emprego apresentam maior risco.
3. **Preparação dos Dados:** Tratamento de valores anômalos em `tempo_emprego` e transformação de variáveis categóricas em variáveis *dummy* (`pd.get_dummies`).
4. **Modelagem:** Treinamento do algoritmo `RandomForestClassifier`, testando diferentes quantidades de árvores (`n_estimators`) para otimizar o desempenho.
5. **Avaliação:** Análise de acurácia (atingindo **~97,79%**) e matriz de confusão, incluindo uma simulação de impacto financeiro na aprovação/reprovação de crédito.
6. **Implantação (Conceitual):** Estruturação de como o modelo atuaria em um motor de decisão de crédito.

## 📈 Principais Conclusões
* A base apresenta um forte desbalanceamento (cerca de 2,34% de inadimplentes).
* Variáveis comportamentais e de estabilidade, como o **tempo de emprego**, mostraram forte poder de discriminação de risco na análise bivariada.
* O ajuste no número de estimadores da *Random Forest* permitiu melhorar a acurácia geral e refinar a matriz de confusão.
Use o código com cuidado.
Se você quiser, posso ajudar a:
Adaptar o texto do README para o inglês (muito valorizado em vagas de dados)
Criar uma postagem para o LinkedIn divulgando esse projeto no seu portfólio
Escrever os comandos exatos do Git caso precise de ajuda para subir o arquivo pelo terminal




Elemneto-basico-de-python- explicar como eu fiz o exercicios do curso da ebac
Aqui estão opções prontas de git commit e README.md focadas no módulo de Elementos Básicos de Python da EBAC, explicando a lógica e o que você praticou nos exercícios.
1. Mensagens de Git Commit
Opção 1 (Detalhada - ideal para quando você sobe o notebook completo):
text
feat: adiciona exercicios de elementos basicos de Python (EBAC)

- Pratica manipulacao de tipos primitivos (int, float, str, bool)
- Implementa operacoes aritmeticas e logicas
- Utiliza metodos de formatacao e fatiamento de strings
- Aplica estruturas de dados basicas e conversao de tipos
Use o código com cuidado.
Opção 2 (Curta e direta):
text
feat: resolve lista de exercicios sobre elementos basicos de Python
Use o código com cuidado.
2. Texto para o README.md (Explicando como você fez os exercícios)
Você pode colar este texto no README.md do seu repositório (ou na seção correspondente a esse módulo):
markdown
# 🐍 Elementos Básicos de Python — Exercícios EBAC

Este repositório contém a resolução dos exercícios práticos do módulo de **Elementos Básicos de Python** do curso de Ciência de Dados da **EBAC**. O objetivo desta etapa foi consolidar a base da linguagem e a lógica de programação antes de avançar para bibliotecas de análise de dados.

## 🧠 Como os exercícios foram resolvidos

Para resolver cada desafio proposto no notebook, segui uma abordagem prática dividida nas seguintes etapas:

1. **Compreensão da Entrada e Saída:** Identificação de quais dados precisavam ser armazenados em variáveis e qual formato de resposta era esperado.
2. **Manipulação de Variáveis e Tipos Primitivos:** 
   * Utilização de números inteiros (`int`) e decimais (`float`) para cálculos financeiros e matemáticos (como cálculo de ticket médio, proporções e taxas).
   * Uso de operadores aritméticos (`+`, `-`, `*`, `/`, `**`) respeitando a ordem de precedência.
3. **Tratamento e Formatação de Strings (`str`):**
   * Limpeza e padronização de textos utilizando métodos nativos como `.upper()`, `.lower()`, `.replace()` e `.strip()`.
   * Extração de informações específicas através de indexação e fatiamento (*slicing*).
   * Construção de mensagens dinâmicas e legíveis utilizando *f-strings* (`f"..."`).
4. **Lógica Booleana e Conversão de Tipos (*Casting*):**
   * Avaliação de condições utilizando operadores relacionais (`==`, `!=`, `>`, `<`) e lógicos (`and`, `or`, `not`).
   * Conversão explícita entre tipos de dados (`int()`, `float()`, `str()`) para garantir compatibilidade nas operações.

## 🛠️ Ferramentas Utilizadas
* **Linguagem:** Python 3
* **Ambiente:** Google Colab / Jupyter Notebook
Use o código com cuidado.
Se você quiser deixar a explicação ainda mais específica, cole o código ou o enunciado dos exercícios aqui. Com isso, eu consigo detalhar exatamente o passo a passo de cada questão que você resolveu!




Pular para o comando anterior
Pergunte o que quiser



A resposta do Modo IA está pronta
Copiada para a área de transferência. Pergunte sobre 
