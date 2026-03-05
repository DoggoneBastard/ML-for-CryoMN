# Literature Review
**Query:** machine learning mdpi  
**Date:** 2026-03-05 10:40:13  
**Max results per source:** 2

## 1. Customer Churn Prediction: A Systematic Review of Recent Advances, Trends, and Challenges in Machine Learning and Deep Learning
**Authors:** Mehdi Imani, M. Joudaki, Ali Beikmohammadi, Hamid Arabnia
**Year:** 2025  
**Citations:** 18  
**Source:** Semantic Scholar  
**DOI:** 10.3390/make7030105  
**URL:** https://www.semanticscholar.org/paper/af444a0ac704922ea674bc7672893fe9d68e62cd  

**Abstract:** Background: Customer churn significantly impacts business revenues. Machine Learning (ML) and Deep Learning (DL) methods are increasingly adopted to predict churn, yet a systematic synthesis of recent advancements is lacking. Objectives: This systematic review evaluates ML and DL approaches for churn prediction, identifying trends, challenges, and research gaps from 2020 to 2024. Data Sources: Six databases (Springer, IEEE, Elsevier, MDPI, ACM, Wiley) were searched via Lens.org for studies published between January 2020 and December 2024. Study Eligibility Criteria: Peer-reviewed original studies applying ML/DL techniques for churn prediction were included. Reviews, preprints, and non-peer-reviewed works were excluded. Methods: Screening followed PRISMA 2020 guidelines. A two-phase strategy identified 240 studies for bibliometric analysis and 61 for detailed qualitative synthesis. Results: Ensemble methods (e.g., XGBoost, LightGBM) remain dominant in ML, while DL approaches (e.g., LSTM, CNN) are increasingly applied to complex data. Challenges include class imbalance, interpretability, concept drift, and limited use of profit-oriented metrics. Explainable AI and adaptive learning show potential but limited real-world adoption. Limitations: No formal risk of bias or certainty assessments were conducted. Study heterogeneity prevented meta-analysis. Conclusions: ML and DL methods have matured as key tools for churn prediction, yet gaps remain in interpretability, real-world deployment, and business-aligned evaluation. Systematic Review Registration: Registered retrospectively in OSF.

---

## 2. A recent advances on autism spectrum disorders in diagnosing based on machine learning and deep learning
**Authors:** Hajir Ammar Hatim, Z. Alyasseri, Norziana Jamil
**Year:** 2025  
**Citations:** 12  
**Source:** Semantic Scholar  
**DOI:** 10.1007/s10462-025-11302-x  
**URL:** https://www.semanticscholar.org/paper/2fac2bb154763298cf530bb955974292e0a97e11  

*No text available.*

---

## 3. Machine Learning and Graph Signal Processing Applied to Healthcare: A Review.
**Authors:** Maria Alice Andrade Calazans, Felipe A B S Ferreira, Fernando A N Santos, Francisco Madeiro, Juliano B Lima
**Year:** 2024  
**Citations:** N/A  
**Source:** PubMed  
**DOI:** 10.3390/bioengineering11070671  
**URL:** https://pubmed.ncbi.nlm.nih.gov/39061753/  

### Full Text
1. IntroductionGraph signal processing (GSP) is an emerging research field, which focuses on generalizing the classical concepts of signal processing in order to expand them to graphs [1]. The need for GSP is related to the considerable amount of information that can be represented as a signal whose samples lie over irregular structures that can be modeled as graphs [1,2]. Among the GSP application scenarios that have attracted the attention of researchers and have been documented in recent studies, one can mention forecasting in the financial market [3], 3D point clouds [4], the Internet of Things (IoT) [5], traffic [6], and sensor, social, physical, and biological networks [7,8,9,10].In the practical use of GSP, machine learning (ML) techniques and, in particular, deep learning (DL) techniques have been playing an important role. This is due to the fact that deep neural networks are adaptable to solving a wide range of problems, providing better or competitive results, when compared to other techniques. The extension of ML to non-Euclidean data gave rise to graph learning (GL) [11] and, consequently, to graph neural networks (GNNs). Such networks have also provided good results in several applications [12,13]. Regarding deep learning on graphs [11], specifically, we can mention the graph convolutional neural networks (GCNNs), in which deep networks with convolutional layers are proposed, such as the operations performed by the traditional convolutional neural network (CNN), but in this case, applied to problems in the non-Euclidean domain, i.e., in graphs [14,15,16].Among the areas that have been highlighted in recent works involving the use of deep learning techniques and graph signal processing, one can mention the medical sciences [17]. Applications of GSP with ML for health have shown growth and been documented in a large number of works published in the literature [18]. In this scope, one identifies papers devoted to applications related to various medical specialties. There is evidence that some of these specialties, such as neurology, for example, have stood out in this context, while other areas are still little explored. The interest of researchers in using GSP and DL in neurology is due to the fact that the human brain can be modeled as a graph, so that its regions can be considered as vertices or nodes and its connectomes at functional and structural levels can be viewed as edges [19,20,21]. Deep networks, on the other hand, are widely used for automatic pattern recognition. In this context, the literature includes works dealing with different objectives, from early diagnosis of Alzheimer’s disease [22] and autism [23] to emotion recognition [24] and imagined speech [25] and multiple sciences [26].In general, scholars in health sciences have demonstrated interest in the development and application of techniques simultaneously based on machine learning, signal processing, and graph theory. The interpretation and analysis of complex irregular data have potential to provide a number of benefits in clinical and hospital practice as an aid in identifying the origin of diseases, the early diagnosis of medical conditions, the verification of possible treatments, and disease prevention [27]. The elements outlined above encouraged us to prepare the present paper, which corresponds to a systematic literature review focusing on machine learning-based healthcare applications, with an emphasis on deep learning applied to signal processing over graphs. The paper presents an overview of the area: the medical specialties with the most papers in GSP in recent years, the ML and GSP techniques that have been most used in healthcare, the most influential authors in the area, and challenges, gaps, and open questions that may provide opportunities for future research. To be more specific, our paper includes the following:A comprehensive overview of ML and GSP applied to healthcare;A panorama of the datasets most used in ML applied to GSP in healthcare and their corresponding description;The identification of gaps, open problems, and promising future research directions in ML applied to GSP in healthcare.The remainder of the paper is organized as follows. In Section 2, the basic fundamentals of graph signal processing, machine learning, and deep learning are presented. Section 3 corresponds to the methodology adopted for the systematic review, such as the scientific databases considered, the search strings used, as well as the inclusion and exclusion criteria of the papers. Section 4 presents the main findings of the review. Section 5 brings a discussion, in which the identified gaps are addressed and future research directions in the area are presented. In Section 6, the final considerations are presented. Figure 1 summarizes the organization of the paper.2. BackgroundIn this section, we provide a concise review of the main concepts related to graph signal processing and machine learning. In the case of GSP, the purpose is to explain what it means to consider a signal in the so-called vertex domain, as well as to indicate the main operators and approaches used in this framework. Regarding machine learning, besides listing the tasks that can be performed with its help and discussing some correlated issues, we highlight aspects of deep learning and the intersections of these tools with graph-based models.2.1. Graph Signal ProcessingGraph signal processing aims to extend classical digital signal-processing methods to signals over irregular domains represented by an arbitrary graph [27,28,29,30].A graph is essentially a set of vertices (nodes) possibly connected by edges. Thus, each sample of a graph signal is associated with a vertex in the corresponding underlying graph; the edge weights reflect the interdependence among the signal samples [30]. In this context, the topology of a graph is inferred or determined according to the proposed application.In terms of orientation, graphs can be directed, if the orientation of the input and output of the edge is considered, or undirected, in the opposite case. Another important characterization concerns the vertex degree. In the case of directed graphs, the vertex degree corresponds to the difference between the weight of edges that depart from it and the weight of edges that arrive at it. The degree of a vertex of an undirected graph, on the other hand, is the sum of the weights of the edges [31,32].Additionally, a graph can be associated with an adjacency matrix, which is denoted by A and contains information about the connectivity of the corresponding graph. If there is an edge connecting vertices vj and vi, the entry Ai,j in the i-th row and the j-column of the referenced matrix is filled with the value of the respective weight; otherwise, Ai,j=0. An adjacency matrix is symmetric if and only if the associated graph is undirected. A graph can also be associated with a degree matrix, which is a diagonal matrix denoted by D and having in the entry Di,i the degree of the vertex vi. Finally, the Laplacian matrix, denoted by L, is obtained by L=D−A [29,32].In the study of graph signal processing, there are two well-established approaches [2]:Spectral graph theory: This is based on the graph Laplacian matrix and considers signals over undirected graphs with real and non-negative weights [1];Algebraic signal processing theory: This considers the adjacency matrix, which assumes the role of the elementary operator. This approach is used in signal analysis of directed and undirected graphs, which may have real or complex weights [30].2.2. Machine Learning and Deep LearningMachine learning corresponds to a subarea of artificial intelligence (AI), which is the field of study of systems that learn problems with examples obtained by training data [33]. Thus, ML aims to propose algorithms that can learn iteratively with the available data, in order to apply such algorithms to automate the construction of models capable of performing classification, regression, and clustering. These tasks can be based, for instance, on decision trees or artificial neural networks [34,35]. The use of such techniques has shown good results for applications in the most diverse areas, including medical diagnosis in health sciences [36,37,38].In ML, two main approaches can be considered: supervised learning, in which training is performed considering labeled data, and the results of the model along the training are compared with the expected (target) outputs; unsupervised learning, in which the model identifies patterns in the data, with typical applications in clustering; in the latter, the data are not labeled and, as a consequence, there is no comparison between the output of the model and the target output along training [39].Deep learning corresponds to a subarea of ML that makes use of deep neural networks. Such networks have high computational complexity and are widely used and disseminated for automatic pattern recognition [40,41,42,43,44,45]. DL techniques have been employed as an effective solution to perform pattern recognition in images, for instance. The most used approach, in this case, employs the so-called CNNs [46]. CNNs operate similarly to the receptive fields of the visual cortex of living beings and are essentially composed of convolutional, pooling, and dense layers [47]. A characteristic of this type of network is its high connectivity, which allows it to process a large amount of input parameters, as required in image processing [48,49].However, CNNs are designed for data with a Euclidean structure. Nevertheless, as previously mentioned, there is a latent need to extend these techniques to the non-Euclidean domain, which can be accomplished by means of their generalization to graphs [50]. This gives rise to graph learning, a field of study that encompasses graph neural networks [51]. Moreover, considering the GL scenario, one has a specific GNN approach, the graph convolutional networks. Analogous to CNNs, GCNs have high connectivity to allow the input of a high number of parameters; in this case, the inputs are graphs [15,52].3. MethodsThe review presented in this paper encompasses papers written in English and published up to 30 October 2023. No starting date was defined for the search of papers in the literature. Four databases of relevance in the field of engineering were used: Science Direct, Institute of Electrical and Electronic Engineers (IEEE Xplore), the Association for Computing Machinery (ACM), and Multidisciplinary Digital Publishing Institute (MDPI).The strings used for the search were as follows:1.“Graph signal processing” AND (COVID OR disease);2.“Graph signal processing” AND (health OR medical OR medicine) AND (“Neural Network” OR “Machine Learning” OR “Deep Learning”).As a result, 396 papers were obtained. Refinements were performed to filter only the relevant papers for the purpose of this review. The first adopted strategy consisted of evaluating the title and the abstract of the papers and discarding those that did not adhere to GSP techniques applied to health. Additionally, repeated papers were also subtracted, so that 50 papers remained for analysis. Finally, 5 more papers were disregarded because they were review papers. As shown in Figure 2, a final sample of 45 papers remained for analysis.It is worth mentioning that five review papers were found, which substantially differ from our paper, both in scope and in selected works, and consequently in their findings.In Khambhati et al. [53], for example, the selected papers concern specifically graphs on dynamic patterns of brain connectivity. In the paper of Dong, Wang, and Abbas [54], the review addresses works in the literature that use deep learning. It is not a review on graph signal processing, although there is a section dedicated to the subject. The paper by Li et al. [55] is a review on graph signal processing and neural networks in the biological data scenario. In this case, despite being a broader review, it is a study more aligned with the biological sciences, since it includes the study of molecules and proteins.The paper by Yin and Kaiser [56] addresses neural flexibility in the human brain. To this end, they reviewed the computational approaches and suggested metrics to classify the flexibility of brain regions. In the work by Yingjie et al. [57], a specific area of the health sciences is analyzed: the work is concerned with the use of deep learning to diagnose liver diseases, and among the methods considered, one observes graph neural networks to detect liver tumors.Unlike the aforementioned papers, our work is in the field of health in general, without a restricted medical area or specialty; we address papers on methods that use machine learning for graph signal processing in health.After the paper selection and exclusion stages, the most relevant characteristics for carrying out the analysis of the 45 selected papers were extracted and synthesized. The information considered in the analysis are those related to the nature and metadata of the paper:Year of publication;First author’s country of affiliation;Studied area.Other issues considered in the analysis were the following:Dataset (size, type, and characteristics of sample);Proposed technique versus the technique used for the comparison;Objective of the study;Performance metrics.The works were analyzed, and gaps and open challenges were identified. The results of such an analysis can serve as guidelines for future work in the area.4. ResultsInitially, lexical analyses were performed on the 45 papers included in this review. The analyses were based on the frequency of occurrence of terms in the titles and the keywords. One of these analyses is the word cloud, which consists of a simplistic visual representation to highlight the words with high recurrence in a previously defined universe [58,59]. Then, the larger the size of the word in the cloud, the more times it occurs in the text. An analysis of this type is depicted in Figure 3, which was obtained to show the co-occurrence of terms in the titles of the papers. In the presented analysis, the terms with the largest font are the most frequent ones in the area under investigation. This study was carried out using the Iramuteq software [60], which is free to use and was developed as open source using the R and Python languages. It can be inferred that terms related to GSP and DL appear very often, as expected, but there is also a considerable occurrence of terms related to neurology, such as: “fmri”, “eeg”, “brain”, and “alzheimer”.One of the encountered issues in the use of word clouds is the lack of grouping of similar terms because of grammatical variations, such as singular and plural [61]. In Iramuteq, this question is solved by the use of textual lemmatization. Thus, a certain level of variation is allowed in the terms, so that they are not considered distinct and the occurrence count is added to its most frequent equivalent term [62,63].Another analysis that can be carried out with the Iramuteq software is the similitude analysis [64], which is based on graph theory. In this case, the most important words in the analysis are represented by vertices of a graph structure and the connections between words correspond to the edges. Thus, it is possible to identify central terms, their connections, and the grouping of words of the same theme just like a hypergraph.Figure 4 shows a similitude analysis obtained by Iramuteq for the titles of the papers. The figure shows a central cluster with words that are frequently related; such a main cluster is connected to other clusters through its secondary terms. As a central term, one observes the word “graph”, as expected. From this, branches are shown with clusters of distinct themes, but originated from the central elements.Figure 5 allows a complementary analysis. In this case, the co-occurrence of keywords is evaluated with the VOSviewer software, which is a tool for the elaboration of bibliometric networks [65]. The most recurrent terms are “graph signal processing”, “deep learning”, “graph learning”, “machine learning”, “fmri”, “connectivity”, “Alzheimer’s disease”, “autism spectrum disorder”, “brain”, “mild cognitive impairment”, and “graph fourier transform”. The nodes were divided into four clusters, so the most frequently related terms are grouped together in the same color. According to the terms shown in the figure, once again as expected, the application focused on neurology is highlighted in the terms in evidence.The distribution of publications by geographic location took into account the country associated with the affiliation of the first author of each paper. This made it possible to analyze the paper distribution by country and by continent, as shown in Figure 6 and Figure 7, respectively. In Figure 6, we verify that there are first authors affiliated with institutions from seventeen different countries, with emphasis on China, the United States of America (USA), Iran, and the United Kingdom; the first two countries have, respectively, ten and seven, and the last two countries have four, of the forty-five first authors.Figure 7 presents the geographical overview from a continental point of view. It can be inferred that there is at least one first author per continent, except in Oceania. The continent with the greatest influence is Asia, corroborating the strong impact provided by China. It is followed by the European continent, which has the United Kingdom and France among the most influential countries according to the number of affiliated first authors. The next continent in this sequence is America, which, despite the strong influence of the USA, has only one other country with two affiliated first authors, Canada. Among the continents with publications, the last is Africa, with only one first author. Europe and Asia together hold 77.8% of first author affiliations.The trend of publications by year was also analyzed in this paper. As illustrated in Figure 8, among the 45 considered papers, the first one was published by Toutain et al. [66], in 2015, being the only paper that year. In 2016, there was again only one publication. In 2017 and 2018, the number increased to two publications per year. In 2020, with eight papers published, the growth was 166.67% compared to the previous year. A growth in the number of publications was observed in 2021, when ten papers were published. In 2022, one observes eleven publications. It can also be inferred that the recent development of the research field that makes use of GSP and ML techniques is evident, which can be observed with the beginning of publications in 2015 and the growth in subsequent years.Figure 8 also presents the number of papers published per year by specialty; it corroborates the emergence of papers that use GSP, ML, and DL for neurology applications, which represents 66.7% of the 45 evaluated studies. However, it is evident that the research field that makes use of GSP and DL techniques is very recent, since the first paper found in this study was published in 2015. On the other hand, it can be said that the area is under consolidation, with the remarkable growth in the number of publications in recent years: in the period from 2020 to 2022, 64.4% of papers were published.Figure 9 shows the eleven areas with publications by means of the tree map, in which the sizes of the squares of the specialties are proportional to the number of publications. Thus, considering the universe of 45 papers selected for this review, neurology is the most prominent (30), followed by genetics (3), cardiology (2), infectology (2), oncology (2), gastroenterology (1), medical clinic (1), cytology (1), psychiatry (1), pneumology (1), and hepatology (1).Figure 10 shows a bar chart of the number of Web of Science citations of the five most cited papers. The paper by Parisot et al. [67] is indicated as the most cited, with 242 citations. Pervaiz et al. [68] ranks second, with 95 citations. There are 38 works that use the study by Sardellitti, Barbarossa, and Lorenzo [69] as a reference, a number reasonably close to the fourth most cited, the work by Hu et al. [70], which has 29 citations. Finally, Zhang et al. [71] ranks fifth, with 22 citations. It can be inferred that there is a considerable difference in the number of citations of [67] compared to the others, which may indicate this work as recommended reading in the area.Figure 11 shows a map of citations obtained with VOSviewer [72]. The map is made up of spheres, labeled with the names of the first authors of the most cited papers and with sizes related to the number of citations received. It is also possible to see the five most cited papers, as shown in the previous figure. In general, the other nodes have similar sizes, indicating that they have received a similar number of citations, reaching a maximum of 21.Figure 12 shows a bibliometric coupling network obtained through analysis in VOSviewer. The nodes of the graphs represent the first authors of the papers, and the size of the vertex is related to the number of citations of the paper. The edges connect the nodes that are bibliographically linked when there is another publication that is cited by the simultaneously linked papers.In order to establish a relationship between each paper and its respective application in the studied health area, Table 1 allocates the 45 evaluated papers to their respective specialty among the eight identified specialties.Table 2 presents the main information extracted from the studied papers. The presented descriptive data refer to the year of publication, the objective of the developed research, the technique proposed in the paper, and with whom it was compared to in order to evaluate its performance. Table 3 presents a set of information on the dataset used in the selected studies, such as the dataset used, the sample size used, and finally, the metric used to evaluate performance.According to Table 3, it is possible to identify the five most used metrics in evaluating the performance of the proposed models, as shown in Figure 13. Accuracy occupies the first place. It is used to assess performance in 26 out of the 45 studies analyzed. In the second place, we observe the F1-score, which appears in 12 works. The AUC holds the third position. It is used in nine papers. In the fourth position, the measures precision and recall are tied. They are used in 8 out of the 45 selected papers. Finally, in the fifth position, we have sensitivity and specificity, which were used in six works.Another analysis obtained from Table 3 concerns the most used databases. Considering the area of neurology, which corresponds to 30 of the 45 articles included in the review, Figure 14 shows that the most used databases were HCP, used 6 times, followed by ADNI with 5 uses, and then ABIDE and DEAP, which were used in 4 papers each. It is important to note that those databases are publicly available.5. DiscussionOne of the challenges reported by the analyzed papers is related to the difficulty of accessing health-related datasets. The limited amount of data (whether images, signals, or medical records, among others) may lead to a lack of generalization of the proposed approaches in the detection or classification of pathologies. Another challenge is related to the reproducibility of research, since different research groups are unable to evaluate new methodological proposals for the reported problem if a common dataset is not available. There is need for more publicly available datasets.Among other limitations addressed by the papers, we can mention data imbalance. In [67], for example, it is mentioned that, in future studies, one of the intentions is to verify the use of graph convolutions to achieve good prediction rates in problems that present data imbalance, since it is considered a factor that hinders the learning of intelligent systems.Regarding the dataset, one possibility to achieve better performances would be to include complementary information to signals or images. This is due to the fact that, in health-related problems, it is relevant to use dataset with a combination of data, such as phenotypic information, because diagnoses may be related to morphological characteristics or conditions and clinical parameters. In [78], for example, it is stated that the work has a limitation because it uses only brain image data; better results could be obtained if the referred additional information would have been employed.Another issue to be considered concerns the medical interpretation of the results obtained by systems using GSP and computational intelligence techniques. Although many proposals achieve good performance, considering the evaluation of objective metrics, which are quite widespread in engineering, it is of paramount importance that there is an understanding of the addressed problem, based on the understanding of what the result means, and also how it impacts the analysis in the health sciences, in order to achieve a broader and more complete analysis for diagnosis. In this context, Valenchon and Coates [80] report their intention that, in further research, rather than the outcome of the proposal indicating whether or not an individual with dementia will progress to Alzheimer’s disease, it presents a mechanism that provides a value of the probability of progression, which guarantees a more complete medical analysis. Another example is [68] in which it is suggested that future work may address possibilities beyond the prediction of clinical diagnosis, one of them being to investigate and suggest possible treatments for the found medical condition.In relation to this, an area that addresses such issues and presents possible solutions to minimize these difficulties is explainable artificial intelligence (XAI) [145]. This is a recent field that concerns the explanations and interpretability necessary in processes that use artificial intelligence techniques in predictions, so that there are justifications for and credibility of the obtained results [146]. In any case, interpretability and explainability are terms that encompass standards and criteria, which must be taken into account according to the associated context [147]. In the case of this review, the context to consider is the medical specialty of the application, and then make the proposal based on computational intelligence understandable to health professionals, knowledgeable about the nature of the problem, reducing the gap between the proposal and clinical practice. In [79], for example, a system for detecting and evaluating signals of neurological examinations for the early detection of Alzheimer’s disease was developed; according to the authors, it would be interesting for health professionals to understand the approach devoted to signal detection, with the aim of enabling the use of programs that validate the proposal in a real medical context.In this context, such issues fall under health 4.0 (H4.0), a term used to relate health advances to industrial technological revolutions. It corresponds to a field that investigates the use of technologies in favor of patient care, based on the use of technology to promote better and faster diagnostic capacity, equipment portability, and greater data management capacity [148]. Thus, the use of technology is aimed at clinical care itself, and can be supported with the use of artificial intelligence, including deep learning techniques for care aimed at early diagnosis, the prevention of the progression of health conditions, and early identification of effective treatments [149,150].An important challenge verified in the analysis of the papers included in the review is the extraction of characteristics from the data used, since this step is not restricted to the extraction itself, but to the selection of more relevant characteristics so that the proposed system for the intended application is able to identify health changes due to the selection of more significant characteristics. In this case, the use of convolutional networks can be considered, since the convolution layers play the role of the extractor of features.Another challenge is the selection of optimal hyperparameters for the proposed techniques, because although they present a good performance, as reported in [22], it is possible to achieve superior results with an assertive selection of hyperparameters. The choice of hyperparameters can be made using grid search, Bayesian optimization, or random search and swarm intelligence.Many applications addressed in the selected papers are in neurology, in which the data evaluated are examinations converted into time series. In this sense, an important analysis to aid diagnosis is to check the regularity of the series and identify noise. That analysis can be carried out by using information theory metrics on graphs, such as permutation entropy and dispersion entropy [151,152]. This could be a promising research area in graph signal processing.Regarding the works in which the specialty of neurology is considered, a frequently considered analysis employs functional connectivity. Therefore, accurate pattern recognition is essential, which can be achieved through the use of robust graph learning techniques, acting in the identification and analysis of connectivity between brain areas. Another possibility of analysis is structural connectivity. According to [75], in future work, a specific structural connectivity for each individual should be considered. This would lead to the definition of multiple spectral domains for brain signals, and would enable the analysis of inter-subject structural variability. Still in this specialty, many studies report the use of atlases to divide the brain into areas. However, there is no consensus on the use of a single atlas to carry out the referred division. It would, therefore, be interesting to verify and test the use of different atlas options for the same dataset, since this choice has a high potential impact on the final classification stage.Due to the good results presented with different techniques that combine GSP and ML, it is possible to use graph neural networks and test the proposed methods in different medical applications of high complexity and that have data available in the literature, as suggested in [66,79]. Considering high-complexity problems, one of the future proposals reported in [95] concerns real-time processing for echocardiogram videos. This could be a major advance in early diagnosis with artificial intelligence, and would represent a significant impact for health sciences.The proposal described in [74], which employs the modified Laplacian matrix to classify attention deficit hyperactivity disorder, presents a promising result, so its use can be considered as an alternative mathematical framework in other medical applications. Likewise, in [103], it is recommended to explore the potential of Multi-GNNs, which consist of combining the characteristics of individual GNNs.An interesting consideration concerns the use of new transforms, because, although the Fourier transform is quite widespread and leads to good results, it is important that different transformation techniques be examined and tested. In [88], for example, the investigation of new transforms is pointed out as a future proposal, as the authors mention the fact that new transformation techniques can lead to improvements in the classification rates.Finally, an important issue is the lack of standardization, so it would be interesting to standardize metrics and evaluation techniques for comparison purposes.6. ConclusionsIn healthcare, GSP has been used to analyze problems related to signals lying in non-Euclidean domains. In addition, ML techniques have been used for pattern recognition and early disease classification and identification. Considering the 45 papers included in the systematic review, 30 of these presented applications for neurology problems, with many of them focused on the diagnosis of cognitive impairment and Alzheimer’s disease. In these cases, most of the data correspond to fMRI and EEG images. However, limitations are reported regarding the number of samples and the number of publicly available dataset.From the presented data regarding the number of publications, it is clear that, despite GSP with ML applied to health being a recent field of study, it has shown an increase in the number of publications, which may indicate an interest of the scientific community in the area. Advances in the scope of GSP with ML in health have attracted the attention of health professionals, since the proposed methods have a high capacity to assist early diagnosis and, consequently, provide speed in decision making by specialists. In any case, there are gaps to be solved, such as a better integration between computational intelligence techniques and clinical practice.This systematic review synthesized the information from selected papers and pointed out the trends of applications that are emerging in the area, as well as methodologies that combine artificial intelligence, graph theory, and health sciences, presenting subsidies for researchers to explore gaps in future work, as well as to reproduce existing work. A limitation of this work is the number of scientific databases considered. Although our study has considered four relevant scientific databases (IEEE Xplore, Science Direct, MDPI, and ACM), it is possible that there are other papers that fit the scope of the review and that have not been included. It is also possible that new papers have been published after the period defined for the inclusion of papers, which was October 2023.In the future, further updates of this literature review can be carried out, including more databases and also revisiting those considered in this paper, since, with the identified trend of publications, there should soon be new research published in the area.

---

## 4. Generative machine learning for de novo drug discovery: A systematic review.
**Authors:** Dominic D Martinelli
**Year:** 2022  
**Citations:** N/A  
**Source:** PubMed  
**DOI:** 10.1016/j.compbiomed.2022.105403  
**URL:** https://pubmed.ncbi.nlm.nih.gov/35339849/  

**Abstract:** Recent research on artificial intelligence indicates that machine learning algorithms can auto-generate novel drug-like molecules. Generative models have revolutionized de novo drug discovery, rendering the explorative process more efficient. Several model frameworks and input formats have been proposed to enhance the performance of intelligent algorithms in generative molecular design. In this systematic literature review of experimental articles and reviews over the last five years, machine learning models, challenges associated with computational molecule design along with proposed solutions, and molecular encoding methods are discussed. A query-based search of the PubMed, ScienceDirect, Springer, Wiley Online Library, arXiv, MDPI, bioRxiv, and IEEE Xplore databases yielded 87 studies. Twelve additional studies were identified via citation searching. Of the articles in which machine learning was implemented, six prominent algorithms were identified: long short-term memory recurrent neural networks (LSTM-RNNs), variational autoencoders (VAEs), generative adversarial networks (GANs), adversarial autoencoders (AAEs), evolutionary algorithms, and gated recurrent unit (GRU-RNNs). Furthermore, eight central challenges were designated: homogeneity of generated molecular libraries, deficient synthesizability, limited assay data, model interpretability, incapacity for multi-property optimization, incomparability, restricted molecule size, and uncertainty in model evaluation. Molecules were encoded either as strings, which were occasionally augmented using randomization, as 2D graphs, or as 3D graphs. Statistical analysis and visualization are performed to illustrate how approaches to machine learning in de novo drug design have evolved over the past five years. Finally, future opportunities and reservations are discussed.

---

## 5. Coronavirus disease (COVID-19) cases analysis using machine-learning applications
**Authors:** Ameer Sardar Kwekha Rashid, Heamn Noori Abduljabbar, Bilal Alhayani
**Year:** 2021  
**Citations:** 343  
**Source:** OpenAlex  
**DOI:** 10.1007/s13204-021-01868-7  
**URL:** https://openalex.org/W3165437720  

### Full Text
Vol.:(0123456789)
1 3
Applied Nanoscience (2023) 13:2013–2025 
https://doi.org/10.1007/s13204-021-01868-7
ORIGINAL ARTICLE
Coronavirus disease (COVID‑19) cases analysis using machine‑learning 
applications
Ameer Sardar Kwekha‑Rashid1 · Heamn N. Abduljabbar2,3 · Bilal Alhayani4
Received: 7 March 2021 / Accepted: 4 May 2021 / Published online: 21 May 2021 
© King Abdulaziz City for Science and Technology 2021
Abstract
Today world thinks about coronavirus disease that which means all even this pandemic disease is not unique. The purpose 
of this study is to detect the role of machine-learning applications and algorithms in investigating and various purposes that 
deals with COVID-19. Review of the studies that had been published during 2020 and were related to this topic by seeking 
in Science Direct, Springer, Hindawi, and MDPI using COVID-19, machine learning, supervised learning, and unsupervised 
learning as keywords. The total articles obtained were 16,306 overall but after limitation; only 14 researches of these articles 
were included in this study. Our findings show that machine learning can produce an important role in COVID-19 investiga-
tions, prediction, and discrimination. In conclusion, machine learning can be involved in the health provider programs and 
plans to assess and triage the COVID-19 cases. Supervised learning showed better results than other Unsupervised learning 
algorithms by having 92.9% testing accuracy. In the future recurrent supervised learning can be utilized for superior accuracy.
Keywords  COVID-19 · Artificial intelligence AI · Machine learning · Machine learning tasks · Supervised and 
un-supervised learning
Introduction
Recently, the world gained rapid progression in tech-
nology and it shows an important role in the developed 
countries. Nowadays all daily life sectors such as edu-
cation, business, marketing, militaries, and communica-
tions, engineering, and health sectors are dependent on 
the new technology applications. The health care center 
is a crucial field that strongly needs to apply the new 
technologies from defining the symptoms to the accu-
rate diagnosis and digital patient’s triage. Coronavirus-2 
(SARSCoV-2) causes severe respiratory infections, and 
respiratory disorders, which results in the novel coro-
navirus disease 2019 (COVID-19) in humans who had 
been reported as the first case in Wuhan city of China in 
December 2019. Later, SARS-CoV-2 was spread world-
wide and transmitted to millions of people and the world 
health organization (WHO) have announced the outbreak 
as a global pandemic since the number of infected people 
is still increasing day by day. As of 16th December 2020, 
the total (global) coronavirus cases were approximately 
73,806,583 with reported deaths of 1,641,635 (Pasupuleti 
et al. 2021). The novel coronavirus appeared in December 
2019, in the Wuhan city of China and the World Health 
Organization (W.H.O) reported it on 31st December 2019. 
The virus produced a global risk and W.H.O named it 
COVID-19 on 11th February 2020 (Wu 2020). Up to the 
present time, there was no specific medication that deals 
directly with this new generation of COVID-19 virus, but 
some of the companies produced several combination 
drugs that basically made up from ethanol, isopropyl alco-
hols, and hydrogen peroxides in different combinations 
 *	 Ameer Sardar Kwekha‑Rashid 
	
ameer.rashid@univsul.edu.iq
	
Heamn N. Abduljabbar 
	
heamn.abduljabbar@su.edu.krd
	
Bilal Alhayani 
	
bilalabed1978@gmail.com
1	
Business Information Technology, College of Administration 
and Economics, University of Sulaimani, Sulaimaniya, Iraq
2	
College of Education, Physics Department, Salahaddin 
University, Shaqlawa, Iraq
3	
Department of radiology and imagingFaculty of Medicine 
and Health Sciences, Universiti Putra Malaysia UPM, 
Seri Kembangan, Malaysia
4	
Electronics and Communication Department, Yildiz 
Technical University, Istanbul, Turkey

2014
	
Applied Nanoscience (2023) 13:2013–2025
1 3
show a significant reaction to the novel virus and had 
been confirmed and accepted by WHO to be used in the 
world (Mahmood et al. 2020). The artificial intelligence 
and deep learning algorithm show the ability to diagnose 
COVID-19 in precise which can be regarded as a sup-
portive factor to improve the common diagnostic meth-
ods including Immunoglobulin M (IgM), Immunoglobulin 
(IgG), chest x-ray, and computed tomography(CT) scan, 
also reverse transcription-polymerase chain reaction (RT-
PCR) and immunochroma to graphic fluorescence assay. 
The developments of a potential technology are one of the 
currently used methods to identify the infection, such as 
a drone with thermal screening without human interven-
tion, which needs to be encouraged (Manigandan 2020). 
The assessment of the research that had been produced 
whether it hits the target of the existing knowledge gaps 
or not can be done by applying an artificialintelligence/
machine learning-based approach to analyze COVID-19 
literature (Doanvo et al. 2020). Thus, the acceleration of 
the diagnosis and treatment of COVID-19 disease is the 
main advantage of these AI-based platforms (Naseem et al. 
2020) which finally shows a huge potential to exponen-
tially enhance and improve health care research (Jamshidi 
et al. 2020). Corona Virus Disease 2019 (COVID-19), 
has become a matter of serious concern for each country 
around the world. AI applications can assist in increas-
ing the accuracy and the speed of identification of cases 
through data mining to deal with the health crisis effi-
ciently, the rapid expansion of the pandemic has created 
huge health care disorders which as a result encouraged 
the real need for immediate reactions to limit the effects. 
Artificial Intelligence shows great applications in deal-
ing with the issue on many sides (Tayarani-N 2020). The 
COVID-19 is an epidemic disease that challenged human 
lives in the world. The systematic reviews showed that 
machine learning ML training algorithms and statistical 
models that are used computers to perform various tasks 
without explicit commands (Bishop 2006). Currently, 
machine learning techniques are used internationally for 
predictions due to their accuracy. However, machine learn-
ing (ML) techniques, have few challenges such as the new 
poor database that is available online. For instance, the 
selection of the appropriate parameters is one of the chal-
lenges involved in training a model or the selection of the 
best Machine learning model for prediction. Depending 
on the available dataset researchers obtained predictions 
by using the best Machine learning model that suits the 
dataset (Shinde 2020). Machine learning techniques can 
be used to extract hidden patterns and data analytics (Khan 
2020). The algorithms of Machine-learning are designed 
for identifying complex patterns and interfaces in the data, 
in the context of unknown and complicated correlation 
patterns among risk factors (Hossain 2019).
Related work
COVID‑19
The contagion disease caused by the SARS-COV-2 virus 
named COVID-19 is requiring extraordinary responses of 
special intensity and possibility to more than 200 countries 
around the world, the first 4 months from its epidemic, the 
number of infected peoples ranged from 2 to 20 million, 
with at least 200,000 deaths. To manage the spread of the 
COVID-19 infection among people rapidly, all the govern-
ments around the world applied severe actions, such as 
the quarantine of hundreds of millions of citizens world-
wide (Alimadadi 2020). Nevertheless, the difficulty of 
distinguishing between the positive and negative COVID-
19 individuals depending on the various symptoms of 
COVID-19, all of these efforts are limited. Therefore, tests 
to detect the SARS–CoV-2 virus are believed to be critical 
to recognize the positive cases of this infection to limit the 
(Brinati et al. 2020). Radiology and imaging are some of 
the most beneficial and critical modalities used for diag-
nosis COVID-19 stage and hazards on the patient’s lungs 
specifically by chest CT scan (Day 2020). Early diagnosis 
of COVID-19 is important to minimize human-to-human 
transmission and patient care. Recently, the separation 
and quarantine of healthy people from the infected or per-
sons who suspect that they are carrying the virus is the 
most effective technique to avoid the spread of COVID-19 
(Deng 2020). Machine-learning techniques role showed 
an important understandings of the COVID-19 diagnosis, 
such as lung computed tomography (CT) scan whether it 
can be regarded as the first screening or an alternative test 
for the real-time inverse transcriptase–polymerase chain 
reaction (RT–PCR), and the differences between COVID-
19 pneumonia and other viral pneumonia using CT scan 
of the lungs(Kassani  et al. (2004)).
Machine learning
Machine learning is one of the most promising tools in 
classification (Hossain 2019). In essence; machine learn-
ing is a model that aims to discover the unknown function, 
dependence, or structure between input and output vari-
ables. Usually, these relations are difficult to be existed 
by explicit algorithms via automated learning process 
(Zhang 2020a). Machine-learning methods are applied to 
predict possible confirmed cases and mortality numbers 
for the upcoming (Hastie et al. 2009). Machine learning 
can be divided into two parts. The first part is to define 
the optimal weight of data fusion of multi-node percep-
tion outcomes and eliminate unusable nodes based on the 

2015
Applied Nanoscience (2023) 13:2013–2025	
1 3
genetic algorithm, while the second part is to find fault 
nodes through a fault recognition neural network (Ünlü 
and Namlı 2020). Machine learning is a subsection of Arti-
ficial Intelligence (AI), and it involves several learning 
paradigms, such as Supervised Learning (SL), Un-super-
vised Learning (UL), and Reinforcement Learning (RL) 
(Shirzadi 2018). Typical ML models consist of classifica-
tion, regression, clustering, anomaly detection, dimension-
ality reduction, and reward maximization (Gao 2020). The 
ML algorithms are trained in the SL paradigm, on labeled 
data sets, meaning that they exist to a ground-truth output 
(continuous or discrete) for every input. Conversely, in UL 
(Bishop (2006)) there is no ground-truth output, and the 
algorithms normally attempt to discover patterns in the 
data. Reinforcement Learning aims to raise the cumula-
tive reward so that it is more suitable for sequential deci-
sion-making tasks (Zhang 2020b). Supervised learning 
has regression and classification; unsupervised learning 
includes cluster analysis and dimensionally reduction, also 
Reinforcement Learning (RL) includes classification and 
control, as illustrated in Fig. 1.
COVID‑19 with machine learning
Recently there are three different perspectives of work 
that had been done on edge computing and the detection 
of (COVID-19) Cases. The viewpoints are including the 
recognizing of (COVID-19) cases by machine-learning 
systems (Table 1). The algorithms for the recognition 
of activity from machine learning and the approaches 
which used in edge computing are considered the Imag-
ing workflows that can inspire machine-learning methods 
that are able of supporting radiologists who search for an 
analysis of complex imaging and text data. For the novel 
COVID-19 there are models capable of analyzing medical 
imaging and recognizing COVID-19 (Shirzadi 2018). Arti-
ficial intelligence AI has various types, machine learning 
(ML), is one of these applications, it had been applied 
successfully to different fields of medicine for detection 
of new genotype–phenotype associations, diagnosis, which 
showed effects on assessment, prediction, diseases classifi-
cation, transcriptomic, and minimizing the death ratio(Gao 
2020).
The technique of automatic classification of COVID-19 
can be applied by comparing general deep learning-based 
feature extraction frameworks to achieve the higher accu-
rate feature, which is an important module of learning, 
MobileNet, DenseNet, Xception, ResNet, InceptionV3, 
InceptionResNetV2, VGGNet, NASNet were selected 
among a group of deep convolutional neural networks 
CNN. The classification then achieved by running the 
extracted features into some of machine-learning classi-
fiers to recognize them as a case of COVID-19 or other 
diseases (Bishop 2006). Progressive machine-learning 
algorithms can integrate and evaluate the extensive data 
that is related to COVID-19 patients to provide best under-
standing of the viral spread pattern, increase the diagnostic 
accuracy, improve fresh, and effective methods of therapy, 
and even can recognize the individuals who, at risk of 
the disease depending on the genetic and physiological 
features (Khanday 2020).
Literature searching strategy and article 
selection
This systematic review paper used articles from online 
digital databases, which include Science Direct, Springer, 
Hindawi, and MDPI databases, two independent authors 
started the search strategy from October 2020 until 
Fig. 1   Overview of machine-
learning types and tasks
Machine Learning Types and Tasks
Un-Supervised Learning
Classification
Supervised Learning
Cluster
Analysis
Dimensionality
reduction
Reinforcement Learning
Classification
Control
Types
yp
Types
Tasks
Tasks
Regression
Table 1   Search strategy and 
paper selection process
Source
After query search
After applying the 
selection criteria
After quality 
assessment
After full 
article read-
ing
Science direct
1254
440
24
4
Springer
5008
1549
217
5
Hindawi
10,000
3027
134
1
MDPI
44
38
20
4
Total
16,306
5054
395
14

2016
	
Applied Nanoscience (2023) 13:2013–2025
1 3
Table 2   Supervised and un-supervised machine learning for analyzing the COVID-19 disease that included articles with the related details of the Dataset, author name, country of publication, 
year of publication, the used method in the study, and their results
n
Author
Year
Country
Dataset
Method
Tasks and Algorithms
Result
1
Khanday et al. (2020)
2020 India
GitHub
212 reports
Supervised learning
Classification
Logistic Regression and Naive 
Bayes
The findings showed that Logis-
tic regression and multinomial 
Nia’’ve Bayes are better than 
the commonly used algorithms 
according to 96% accuracy 
obtained from the findings
2
Burdick et al. (2020a)
2020 USA
United States health systems
197 patients
Supervised learning
Classification
Logistic Regression
Their results showed that this algo-
rithm displays a higher diagnostic 
odds ratio (12.58) for foreseeing 
ventilation and effectively triage 
patients than a comparator early 
warning system, such as Modified 
Early Warning Score (MEWS) 
which showed (0.78) sensitivity, 
while this algorithm showed (0.90) 
sensitivity which leads to higher 
specificity (p < 0.05), also it shows 
the capability of accurate identifi-
cation 16% of patients more than 
a commonly used scoring system 
which results in minimizing false-
positive results
3
Varun et al. (2020)
2020 USA
184,319 reported cases
Supervised learning
Classifications
Convolutional Neural Networks 
(CNN)
In response to this crisis, the medi-
cal and academic centers in New 
York City issued a call to action to 
artificial intelligence researchers to 
leverage their electronic medi-
cal record (EMR) data to better 
understand SARS-COV-2 patients. 
Due to the scarcity of ventilators 
and a reported need for a quick 
an accurate method of triaging 
patients at risk for respiratory 
failure, our purpose was to develop 
a machine-learning algorithm 
for frontline physicians in the 
emergency department and the 
inpatient floors to better risk-assess 
patients and predict who would 
require intubation and mechanical 
ventilation

2017
Applied Nanoscience (2023) 13:2013–2025	
1 3
Table 2   (continued)
n
Author
Year
Country
Dataset
Method
Tasks and Algorithms
Result
4
Luca et al. (2020)
2020 Italy
85 chest X-rays
Supervised Learning
Classification
K-nearest neighbors classifier 
(k-NN)
In the paper, we propose a method 
aimed to automatically detect the 
COVID-19 disease by analyz-
ing medical images. We exploit 
supervised machine-learning 
techniques building a model con-
sidering a data set freely available 
for research purposes of 85 chest 
X-rays. The experiment shows 
the effectiveness of the proposed 
method in the discrimination 
between the COVID-19 disease 
and other pulmonary diseases
5
Constantin et al. (2020)
2020 Germany
152 datasets of COVID-19 
patients, 500 chest CTs
Supervised learning
Classifications
Convolutional Neural Network 
(CNN)
The findings showed that the com-
bining between machine learning 
and a clinically embedded software 
developed platform allowed time-
efficient development, immediate 
deployment, and fast adoption 
in medical routine. Finally they 
achieved the algorithm for fully 
automated segmentation of the 
lung and opacity quantification 
within just 10 days was ready 
for medical use and achieved 
human-level performance even for 
complex cases
6
Lamiaa et al. ( 2020)
2020 Egypt
COVID-19 5000 cases
Supervised learning
Regression
Linear Regression model
The result showed that the desig-
nated models, such as the expo-
nential, fourth-degree, fifth-degree, 
and sixth-degree polynomial 
regression models are brilliant 
especially the fourth-degree model 
which will benefit the govern-
ment to prepare their procedures 
for 1 month. Furthermore, they 
introduced a well-known log that 
will grow up the regression model 
and will result in obtaining the epi-
demic peak and the last time of the 
epidemic during a specific time in 
2020. Besides, the final report of 
the total size of COVID-19 cases

2018
	
Applied Nanoscience (2023) 13:2013–2025
1 3
Table 2   (continued)
n
Author
Year
Country
Dataset
Method
Tasks and Algorithms
Result
7
Dan  et al. (2020)
2020 Israel
6995 patients in Sheba Medical 
Center
Supervised learning
Classifications
Artificial Neural Network (ANN)
The most contributory variables 
to the models were APACHE II 
score, white blood cell count, and 
time from symptoms to admission, 
oxygen saturation, and blood lym-
phocytes count. Machine-learning 
models demonstrated high efficacy 
in predicting critical COVID-19 
compared to the most efficacious 
tools available. Hence, artificial 
intelligence may be applied for 
accurate risk prediction of patients 
with COVID-19, to optimize 
patients triage and in-hospital 
allocation, better prioritization of 
medical resources, and improved 
overall management of the 
COVID-19 pandemic
8
Joep et al. (2020)
2020 Netherlands
319 patients
Supervised learning
Classification
Logistic regression
Chest CT, using the CO-RADS 
scoring system, is a sensitive and 
specific method that can aid in the 
diagnosis of COVID-19, especially 
if RT–PCR tests are scarce during 
an outbreak. Combining a predic-
tive machine-learning model could 
further improve the accuracy of 
diagnostic chest CT for COVID-
19. Further candidate predictors 
should be analyzed to improve our 
model. However, RT–PCR should 
remain the primary standard of 
testing as up to 9% of RT–PCR 
positive patients are not diagnosed 
by chest CT or our machine-learn-
ing model

2019
Applied Nanoscience (2023) 13:2013–2025	
1 3
Table 2   (continued)
n
Author
Year
Country
Dataset
Method
Tasks and Algorithms
Result
9
Christopher et al. (2020)
2020 Germany
368 independent variables
Supervised learning
Classifications
Naive Bayes
They focused on variables and fac-
tors that increase the COVID-19 
incidence in Germany depending 
on the multi-method ESDA tactic 
which provides a unique insight 
into spatial and spatial non-station-
aries of COVID-19 occurrence, the 
variables, such as built environ-
ment densities, infrastructure, and 
socioeconomic characteristics all 
showed an association with inci-
dence of COVID-19 in Germany 
after assessment by the county 
scale
Their research outcome suggests that 
implementation social distancing 
and reducing needless travel can 
be important methods for reducing 
contamination
10 Hoyt et al. (2020b)
2020 U.S
290 patients
Supervised learning
Classification Logistic Regression
The findings showed that there is no 
correlation between the mortality 
and treatment in the entire popula-
tion as the hydroxychloroquine 
was associated with a statistically 
significant (p = 0.011) rise in 
survival the adjusted hazard ratio 
was 0.29, 95% with a confidence 
interval (CI) 0.11–0.75. Although 
the patients who were indicted 
by the algorithm the adjusted 
survival was 82.6% in the treated 
group and 51.2% in the group who 
were not treated, after machine-
learning applications the algorithm 
detected 31% of improving among 
the COVID-19 population which 
shows the important role of the 
machine-learning application in 
medicine

2020
	
Applied Nanoscience (2023) 13:2013–2025
1 3
Table 2   (continued)
n
Author
Year
Country
Dataset
Method
Tasks and Algorithms
Result
11 María.et al. ( et al. 2020) 2020 International
Food for each of the 170 countries
Unsupervised learning Clustering
K-means clustering
The research findings stated that 
countries with the highest death 
ratio were those who had a high 
consumption of fats, while coun-
tries with a lower death rate have a 
higher level of cereal consumption 
followed by a lower total average 
intake of kilocalories
12 Shinwoo et al. (2020)
2020 U.S.A
790 Korean immigrants
Supervised learning
Classifications
Artificial Neural Network (ANN)
Their result showed The Artificial 
Neural Network (ANN) analysis, 
which is a statistical model and 
able to examine complex non-lin-
ear interactions of variables, was 
applied. The algorithm perfectly 
predicted the person’s flexibility, 
familiarities of everyday discern-
ments, and the racism actions 
toward Asians in the U.S. since 
the beginning of the COVID-19 
pandemic which finally provides 
important suggestions for public 
health practitioners (Zhang 2020b)
13 Yigrem.et al. (2020)
2020 Southern Ethiopia 244 samples
Supervised learning
Classification Logistic Regression
Results showed that more than half 
of the research participants were 
presented with perceived stress of 
coronavirus disease, which means 
that there is a strong correlation 
between the health care staff and 
perceived stress of COVID-19
14 Abolfazl et al. (2020)
2020 USA
US Centers for Disease and Con-
trol and Johns Hopkins Univer-
sity. Database of 57 candidate
Supervised learning
Classification Artificial Neural 
Networks (ANN)
Results showed that the presented 
model (logistic regression) shown 
that these factors and variables 
describe the presence/absence 
of the hotspot of the COVID-19 
incidence which was clarified 
by Getis-Ord Gi (p < 0.05) in a 
geographic information system. 
As a result, the findings provided 
valuable insights for public health 
decision makers in categorizing 
the effect of the potential risk 
factors associated with COVID-19 
incidence level

2021
Applied Nanoscience (2023) 13:2013–2025	
1 3
December 2020. The used keywords were “COVID-19; 
Machine Learning; Supervised Learning; Un-supervised 
Learning.’’ They were connected to the relevant articles 
using “and’’, or “or’’ to find the studies that deals with 
human disease and COVID-19. The total number of the stud-
ies were (16,306) articles from all the databases, accord-
ing to the inclusion and exclusion criteria this number was 
limited. The limitation includes selecting the publication 
year (2019–2021), the articles type original articles that had 
been published as journal articles in English language only 
included. This selection strategy reduced the total number 
to 5054 articles, then after quality assessment of these stud-
ies there was 395 articles which remained, then finally the 
full text article reading minimized the last included articles 
to 14. The included articles are presented according to the 
author’s name, publication’s year, country, the used dataset, 
the applied method, and finally their results in (Table 2).
Machine‑learning types applied
According to Fig. 2, supervised learning is the dominant 
machine-learning type applied for production lines. The 
majority of studies used both supervised learning methods 
which were (92.9%), whereas unsupervised learning was 
(7.1%).
Results
Machine‑learning tasks addressed
Figure 3 shows that classification is the main task, which 
accounts for about (86%) of all selected papers. There are 
about (7%) of papers that applied for each of the regression 
and clustering.
Machine‑learning algorithms used
Figure 4 shows that the logistic regression is largely applied 
in production lines. Logistic regression is the most fre-
quently applied machine-learning algorithm, including five 
papers in 14 papers. Artificial neural network algorithm 
(ANN) and CNN (convolutional neural network) are in the 
second and third ranks which were three and two papers in 
14 papers, respectively. Linear regression, K-Means, KNN 
(K-nearest neighbors), and Naive Bayes are the other algo-
rithms applied for production lines.
Discussions and implications
The new transmitted virus was discovered and spread out 
from Wuhan city of China in December 2019 and affected 
more than (100) countries around the world in a very short 
time (Wu 2020). It was represented and introduced to the 
World Health Organization (W.H.O) on 31st December 
2019. The virus was then termed COVID-19 by W.H.O on 
11th February 2020, because it formed a global risk (Wu 
Fig. 2   Distribution of machine-learning types
Fig. 3   Distribution of machine-learning tasks
Fig. 4   Distribution of machine-learning algorithms

2022
	
Applied Nanoscience (2023) 13:2013–2025
1 3
2020). This family of viruses also includes SARS, ARDS. 
W.H.O confirmed this eruption as a public health emergency 
(Manigandan et al. 2020). Technology progressions have a 
fast effect on each field of life; the medical field is one of 
the important direct daily related to people’s lives. Recently 
Artificial intelligence AI had been introduced to the medical 
field and it has shown promising outcomes in health care due 
to the high accuracy of data analysis which makes an exact 
decision making. Researchers all over the world tried to find 
a method to improve the clinical diagnosis and minimize the 
rapid spread of this virus so that they involved AI algorithms 
in the diagnosis of this disease. This review paper explains 
various AI algorithms that people used in their researches 
and will compare their results to demonstrate the best accu-
rate method that shows the most improving in COVID-19 
diagnosis. The total studies that used in this research are (14) 
original articles, all of them used supervised learning as the 
main method, but the algorithms were differed among them 
according to the research purpose.
A study recently published 2020 in India they extracted 
their dataset from GitHub which was 212 reports of 1000 
cases, they used supervised learning as their main method 
in machine-learning application, and the algorithm that they 
applied was classification logistic regression and multinomi-
nal Nia’’ve Bayes. The findings showed that Logistic regres-
sion and multinominal Nia’’ve Bayes are better than the 
commonly used algorithms according to 96% accuracy 
obtained from the findings (Khanday 2020). Scientists in the 
USA published an article 2020 they relied on United States 
health systems to custom 197 patients as their data, the main 
method that they used was supervised learning, while the 
algorithm was classification logistic regression, their results 
showed that this algorithm displays higher diagnostic odds 
ratio (12.58) for foreseeing ventilation and effectively triage 
patients than a comparator early warning system, such as 
Modified Early Warning Score (MEWS) which showed 
(0.78) sensitivity, while this algorithm showed (0.90) sensi-
tivity which leads to higher specificity (p < 0.05), also it 
shows the capability of accurate identification 16% of 
patients more than a commonly used scoring system which 
results in minimizing false-positive results (Burdick 2020a). 
Varun et al. (2020) used 184,319 reported cases as a ataset 
in his article in which he applied the same method super-
vised learning but with a different algorithm which was con-
volutional neural network CNN and their outcomes were in 
response to this crisis, the medical and academic centers in 
New York City issued a call to action to artificial intelligence 
researchers to leverage their electronic medical record 
(EMR) data to better understand SARS-COV-2 patients. Due 
to the scarcity of ventilators and a reported need for a quick 
and accurate method of triaging patients at risk for respira-
tory failure, our purpose was to develop a machine-learning 
algorithm for frontline physicians in the emergency 
department and the inpatient floors to better risk-assess 
patients and predict who would require intubation and 
mechanical ventilation (Arvind 2020). Meanwhile, another 
study had been published in Italy by (Luca et al. 2020) who 
used also supervised learning in their methodology but they 
used a different algorithm this time called K-nearest neigh-
bors classifier K-NN, their research results showed that the 
proposed method that aims to detect the COVID-19 disease 
by analyzing medical images by building a model allowing 
an easily data set availability for research purposes using 85 
chest X-rays. The research shows the effectiveness of the 
proposed method in the discrimination between the COVID-
19 disease and other pulmonary diseases (Brunese 2020). 
Constantin et al. (2020) published an article in Germany he 
depended on 152 datasets of COVID-19 patients and 500 
chest CT scans, he also relied on supervised learning but 
using Neural Network Algorithm for analyzing these data. 
Their findings showed that the combining between machine 
learning and a clinically embedded software developed plat-
form allowed time-efficient development, immediate deploy-
ment, and fast adoption in medical routine. Finally, they 
achieved the algorithm for fully automated segmentation of 
the lung, and opacity quantification within just 10 days was 
ready for medical use and achieved human-level perfor-
mance even for complex cases (Anastasopoulos 2020). Far 
away from Europe and the USA, a study conducted by 
(Amar et al. 2020) in Egypt depended on 5000 COVID-19 
cases as a dataset. They had chosen supervised learning as 
their method than using regression analysis as the selected 
algorithm. The result showed that the designated models, 
such as the exponential, fourth-degree, fifth-degree, and 
sixth-degree polynomial regression models are brilliant 
especially the fourth-degree model which will benefit the 
government to prepare their procedures for 1 month. Fur-
thermore, they introduced a well-known log that will grow 
up the regression model and will result in obtaining the epi-
demic peak and the last time of the epidemic during a spe-
cific time in 2020. Besides, the final report of the total size 
of COVID-19 cases (Amar et al. 2020). Researchers in Israel 
presented research by (Dan et al. 2020) they extracted 6995 
patient reports from Sheba Medical Center to be used as 
research data, they also used supervised learning as the main 
method, and then they selected the artificial neural network 
ANN as the used algorithm in their study, depending on the 
patient biography it had been demonstrated that APACHE 
II score, white blood cell WBC count, duration from symp-
toms to admission, oxygen saturation and blood lymphocytes 
count were the most related variables to the used models. 
The findings demonstrated that Machine-learning (ML) 
models showed high efficiency in predicting serious COVID-
19 as compared to the other efficient tools available. Here-
after, the results suggested artificial intelligence be applied 
for accurate risk estimation of COVID-19 patients, to 

2023
Applied Nanoscience (2023) 13:2013–2025	
1 3
enhance patient triage (Assaf 2020). In a study conducted 
by (Hermans et al. 2020) in the Netherlands, their article 
used 319 patients as the dataset and they selected supervised 
learning as their method, while the logistic regression was 
the selected algorithm. In this article, they depended on the 
patient’s chest CT scan scores, and the RT–PCR test the 
results demonstrated that Chest CT, using the CO-RADS 
scoring system, is a specific useful method that can lead to 
accurate diagnosis of COVID-19, particularly if RT–PCR 
tests are uncommon during an epidemic. Also merging a 
predictive machine-learning model may more improve the 
diagnosis accuracy of chest CT scans for COVID-19 
patients. Nevertheless, they recommended RT–PCR must 
remain as the primary standard of testing, because up to 9% 
of patients with positive RT–PCR were not identified by 
chest CT or the presented machine-learning model (Hermans 
2020). In Germany, Christopher et al. (2020) used 368 inde-
pendent variables as a sample size in their article which built 
its methodology on supervised learning, and the model was 
Bayesian machine-learning analysis. They focused on vari-
ables and factors that increase the COVID-19 incidence in 
Germany depending on the multi-method ESDA tactic 
which provides a unique insight into spatial and spatial non-
stationaries of COVID-19 occurrence, the variables, such as 
built environment densities, infrastructure, and socioeco-
nomic characteristics all showed an association with inci-
dence of COVID-19 in Germany after assessment by the 
county scale. Their research outcome suggests that imple-
mentation social distancing and reducing needless travel can 
be important methods for reducing contamination (Scarpone 
2020).  Hoyt et al. (2020b) presented an article that depended 
on the data obtained from 290 patients to use supervised 
learning in their article and the logistic regression as the 
specific algorithm, to find the correlation between the treat-
ment and the mortality in the entire 290 population that is 
infected by COVID-19 in the USA by detecting the hazards 
on the entire population the 290 patients who enrolled in 
their research and also on the subpopulation who prepared 
for the suitable treatment identified by the algorithm. The 
findings showed that there is no correlation between the 
mortality and treatment in the entire population as the 
hydroxychloroquine was associated with a statistically sig-
nificant (p = 0.011) rise in survival the adjusted hazard ratio 
was 0.29, 95% with a confidence interval (CI) 0.11–0.75. 
Although the patients who were indicted by the algorithm 
the adjusted survival was 82.6% in the treated group and 
51.2% in the group who were not treated, after machine-
learning applications the algorithm detected 31% of improv-
ing among the COVID-19 population which shows the 
important role of the machine-learning application in medi-
cine (Burdick 2020b). Reichberg et al. (2020) used the inter-
national program food for 170 countries as a source of their 
research using unsupervised learning and specifically the 
K-means clustering algorithm to find the association 
between obesity and mortality in the COVID-19 countries.
The research findings stated that countries with the high-
est death ratio were those who had a high consumption of 
fats, while countries with a lower death rate have a higher 
level of cereal consumption followed by a lower total aver-
age intake of kilocalories (García-Ordás, et al. (2020)). A 
study conducted to (Shinwoo et al. 2020) their research 
data were extracted from the immigrant Korean COVID-
19 patients who were 290 cases from 12 states all of them 
older than 18 years, the study observed the ability to the 
prediction of discrimination-related variables, such as rac-
ism effects, and sociodemographic factors that influence the 
psychological distress level during the COVID-19 pandemic, 
they nominated the supervised learning as the method and 
then using the Artificial Neural Network ANN as the main 
algorithm, their result showed The Artificial Neural Net-
work (ANN) analysis, which is a statistical model and able 
to examine complex non-linear interactions of variables, was 
applied. The algorithm perfectly predicted the person’s flex-
ibility, familiarities of everyday discernments, and the racist 
actions toward Asians in the U.S. since the beginning of the 
COVID-19 pandemic which finally provides important sug-
gestions for public health practitioners (Choi 2020). During 
the same time, a study presented by (Yigrem et al. 2020) 
conducted a cross-study based on 244 of the healthcare pro-
viders in Dilla, Southern Ethiopia. Supervised learning was 
used in the methodology and then they analyzed the data by 
logistic regression algorithm to find the association between 
the perceived stress of COVID-19 and the health care pro-
viders. Results showed that more than half of the research 
participants were presented with perceived stress of corona-
virus disease, which means that there is a strong correlation 
between the health care staff and perceived stress of COVID-
19 (Chekole, et al. (2020)). Finally, the last article conducted 
by (Abolfazl et al. 2020) their study used 57 samples of 
COVID-19 cases from the USA to find out the relationship 
between the sociodemographic and environmental variables, 
other diseases, such as chronic heart disease, leukemia, 
and pancreatic cancer, also socioeconomic factors and the 
death ratio due to COVID-19 disease. Results showed that 
the presented model (logistic regression) shown that these 
factors and variables describe the presence/absence of the 
hotspot of the COVID-19 incidence which was clarified by 
Getis-Ord Gi (p < 0.05) in a geographic information system. 
As a result, the findings provided valuable insights for pub-
lic health decision makers in categorizing the effect of the 
potential risk factors associated with COVID-19 incidence 
level (Mollalo et al. 2020).

2024
	
Applied Nanoscience (2023) 13:2013–2025
1 3
Conclusion
This study focused on the articles that applied machine-
learning applications in COVID-19 disease for various pur-
poses with different algorithms, 14 from 16 articles used 
supervised learning, and only one among them used unsu-
pervised learning another one used both methods supervised 
and unsupervised learning and both of them shows accu-
rate results. The studies used different machine-learning 
algorithms in different countries and by different authors 
but all of them related to the COVID-19 pandemic, (5) of 
these articles used Logistic regression algorithm, and all 
of them showed promising results in the COVID-19 health 
care applications and involvement. While (3) of the arti-
cles used artificial neural network (ANN) which also shows 
successful results, the rest of the 14 articles used different 
supervised and unsupervised learning algorithms and all of 
the models showed accurate results. Our conclusion is ML 
applications in medicine showed promising results with high 
accuracy, sensitivity, and specificity using different models 
and algorithms. In general, the paper results explored the 
supervised learning is more accurate to detect the COVID-
19 cases which were above (92%) compare to the unsuper-
vised learning which was mere (7.1%).
Funding  This study was self- funded.
Declarations 
Conflict of interest  There are no conflicts of interest.
References
Alimadadi A et al (2020) Artificial intelligence and machine learning 
to fight COVID-19. American Physiological Society, Bethesda
Amar LA, Taha AA, Mohamed MY (2020) Prediction of the final size 
for COVID-19 epidemic using machine learning: a case study of 
Egypt. Infect Dis Model 5:622–634
Anastasopoulos C et al (2020) Development and clinical implementa-
tion of tailored image analysis tools for COVID-19 in the midst of 
the pandemic: the synergetic effect of an open, clinically embed-
ded software development platform and machine learning. Eur J 
Radiol 131:109233
Arvind V et al (2020) Development of a machine learning algorithm to 
predict intubation among hospitalized patients with COVID-19. 
J Crit Care 62:25–30
Assaf D et al (2020) Utilization of machine-learning models to accu-
rately predict the risk for critical COVID-19. Intern Emerg Med 
15(8):1435–1443
Bishop CM (2006) Pattern recognition and machine learning. Springer, 
Berlin
Brinati D et al (2020) Detection of COVID-19 infection from routine 
blood exams with machine learning: a feasibility study. J Med 
Syst 44:135
Brunese L et al (2020) Machine learning for coronavirus COVID-19 
detection from chest x-rays. Proced Comput Sci 176:2212–2221
Burdick H et al (2020a) Prediction of respiratory decompensation 
in Covid-19 patients using machine learning: the READY trial. 
Comput Biol Med 124:103949
Burdick H et al (2020b) Is machine learning a better way to identify 
COVID-19 patients who might benefit from hydroxychloroquine 
treatment? The Identify Trial. J Clin Med 9(12):3834
Chekole YA et al (2020) Perceived Stress and Its Associated Factors 
during COVID-19 among Healthcare Providers in Ethiopia: a 
cross-sectional study. Adv Public Health. https://​doi.​org/​10.​1155/​
2020/​50368​61
Choi S et al (2020) Predicting psychological distress amid the COVID-
19 pandemic by machine learning: discrimination and coping 
mechanisms of Korean Immigrants in the US. Int J Environ Res 
Public Health 17(17):6057
Day M (2020) Covid-19: identifying and isolating asymptomatic peo-
ple helped eliminate virus in Italian village. BMJ 368:135
Deng X et al (2020) A classification–detection approach of COVID-
19 based on chest X-ray and CT by using keras pre-trained deep 
learning models. Comput Model Eng Sci 125(2):579–596
Doanvo A et al (2020) Machine learning maps research needs in covid-
19 literature. Patterns 1(9):100–123
Gao K et al (2020) Julia language in machine learning: algorithms, 
applications, and open issues. Comput Sci Rev 37:100254
García-Ordás MT et al (2020) Evaluation of country dietary habits 
using machine learning techniques in relation to deaths from 
COVID-19. Healthcare 8:371
Hastie TR, Tibshirani JF (2009) The elements of statistical learning: 
data mining, inference, and prediction. Springer Science and Busi-
ness Media, Berlin
Hermans JJ et al (2020) Chest CT for triage during COVID-19 on 
the emergency department: myth or truth? Emerg Radiol 
27(6):641–651
Hossain B et al (2019) Surgical outcome prediction in total knee 
arthroplasty using machine learning. Intell Autom Soft Comput 
25(1):105–115
Jamshidi M et al (2020) Artificial intelligence and COVID-19: deep 
learning approaches for diagnosis and treatment. IEEE Access 
8:109581–109595
Kassani SH et al (2020) Automatic detection of coronavirus disease 
(COVID-19) in X-ray and CT images: a machine learning-based 
approach 10(4):1–18
Khan MA et al (2020) Intelligent cloud based heart disease predic-
tion system empowered with supervised machine learning. CMC 
Comput Mater Cont 65(1):139–151
Khanday AMUD et al (2020) Machine learning based approaches for 
detecting COVID-19 using clinical text data. Int J Inf Technol 
12(3):731–739
Mahmood A et al (2020) COVID-19 and frequent use of hand sanitiz-
ers; human health and environmental hazards by exposure path-
ways. Sci Total Environ 742(44):140561
Manigandan S, Ming-Tsang W, Vinoth KP, Vinay BR, Arivalagan P, 
Kathirvel B (2020) A systematic review on recent trends in trans-
mission, diagnosis, prevention and imaging features of COVID-
19. Process Biochem 98(11):233–240. https://​doi.​org/​10.​1016/j.​
procb​io.​2020.​08.​016
Mollalo A, Rivera KM, Vahedi B (2020) Artificial neural network mod-
eling of novel coronavirus (COVID-19) incidence rates across 
the continental United States. Int J Environ Res Public Health 
17(12):4204
Naseem M et al (2020) Exploring the potential of artificial intelligence 
and machine learning to combat COVID-19 and existing oppor-
tunities for LMIC: a Scoping review. J Primary Care Community 
Health 11:1–11

2025
Applied Nanoscience (2023) 13:2013–2025	
1 3
Pasupuleti RR et al (2021) Rapid determination of remdesivir (SARS-
CoV-2 drug) in human plasma for therapeutic drug monitoring in 
COVID-19-Patients. Process Biochem 102(3):150–156
Reichberg SB, Mitra PP, Haghamad A, Ramrattan G, Crawford JM, 
Northwell COVID-19 Research Consortium et al (2020) Rapid 
emergence of SARS-CoV-2 in the greater New York metropolitan 
area: geolocation, demographics, positivity rates, and hospitaliza-
tion for 46 793 persons tested by northwell health. Clin Infect Dis 
71(12):3204–3213
Scarpone C et al (2020) A multimethod approach for county-scale geo-
spatial analysis of emerging infectious diseases: a cross-sectional 
case study of COVID-19 incidence in Germany. Int J Health 
Geogr 19(1):1–17
Shinde GR et al (2020) Forecasting models for coronavirus disease 
(COVID-19): a survey of the state-of-the-art. SN Comput Sci 
1(4):1–15
Shirzadi A et al (2018) Novel GIS based machine learning algorithms 
for shallow landslide susceptibility mapping. Sensors 18(11):3777
Tayarani-N, M.-H (2020) Applications of artificial intelligence in bat-
tling against Covid-19: a literature review. Chaos Solitons Fractals 
142:110338
Ünlü R, Namlı E (2020) Machine learning and classical forecasting 
methods based decision support systems for COVID-19. CMC 
Comput Mater Cont 64(3):1383–1399
Wu F et al (2020) A new coronavirus associated with human respira-
tory disease in China. Nature 579(7798):265–269
Zhang C et al (2020a) Applying feature-weighted gradient decent 
k-nearest neighbor to select promising projects for scientific fund-
ing. CMC Comput Mater Cont 64(3):1741–1753
Zhang Y et al (2020b) Overview on routing and resource allocation 
based machine learning in optical networks. Opt Fiber Technol 
60:102355
Publisher’s Note  Springer Nature remains neutral with regard to 
jurisdictional claims in published maps and institutional affiliations.

---

## 6. State of Industry 5.0—Analysis and Identification of Current Research Trends
**Authors:** Aditya Akundi, Daniel Euresti, Sergio Luna, Wilma Ankobiah, Amit Lopes, Immanuel Edinbarough
**Year:** 2022  
**Citations:** 415  
**Source:** OpenAlex  
**DOI:** 10.3390/asi5010027  
**URL:** https://openalex.org/W4212802598  

### Full Text


Citation: Akundi, A.; Euresti, D.;
Luna, S.; Ankobiah, W.; Lopes, A.;
Edinbarough, I. State of Industry
5.0—Analysis and Identiﬁcation of
Current Research Trends. Appl. Syst.
Innov. 2022, 5, 27. https://doi.org/
10.3390/asi5010027
Academic Editor: Mario Di Nardo
Received: 30 December 2021
Accepted: 11 February 2022
Published: 17 February 2022
Publisher’s Note: MDPI stays neutral
with regard to jurisdictional claims in
published maps and institutional afﬁl-
iations.
Copyright:
© 2022 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed
under
the
terms
and
conditions of the Creative Commons
Attribution (CC BY) license (https://
creativecommons.org/licenses/by/
4.0/).
Article
State of Industry 5.0—Analysis and Identiﬁcation of Current
Research Trends
Aditya Akundi 1,*, Daniel Euresti 1, Sergio Luna 2, Wilma Ankobiah 3, Amit Lopes 2 and Immanuel Edinbarough 1
1
Complex Engineering Systems Laboratory, Department of Informatics and Engineering Systems,
The University of Texas Rio Grande Valley, Brownsville, TX 78520, USA; daniel.euresti01@utrgv.edu (D.E.);
immanuel.edinbarough@utrgv.edu (I.E.)
2
Industrial Manufacturing and Systems Engineering Department, The University of Texas at El Paso,
El Paso, TX 79968, USA; salunafong@utep.edu (S.L.); ajlopes@utep.edu (A.L.)
3
Complex Engineering Systems Laboratory, Department of Manufacturing and Industrial Engineering,
The University of Texas Rio Grande Valley, Brownsville, TX 78520, USA; wilma.ankobiah01@utrgv.edu
*
Correspondence: satya.akundi@utrgv.edu
Abstract: The term Industry 4.0, coined to be the fourth industrial revolution, refers to a higher level
of automation for operational productivity and efﬁciency by connecting virtual and physical worlds
in an industry. With Industry 4.0 being unable to address and meet increased drive of personalization,
the term Industry 5.0 was coined for addressing personalized manufacturing and empowering
humans in manufacturing processes. The onset of the term Industry 5.0 is observed to have various
views of how it is deﬁned and what constitutes the reconciliation between humans and machines.
This serves as the motivation of this paper in identifying and analyzing the various themes and
research trends of what Industry 5.0 is using text mining tools and techniques. Toward this, the
abstracts of 196 published papers based on the keyword “Industry 5.0” search in IEEE, science direct
and MDPI data bases were extracted. Data cleaning and preprocessing were performed for further
analysis to apply text mining techniques of key terms extraction and frequency analysis. Further
topic mining i.e., unsupervised machine learning method was used for exploring the data. It is
observed that the terms artiﬁcial intelligence (AI), big data, supply chain, digital transformation,
machine learning, internet of things (IoT), are among the most often used and among several enablers
that have been identiﬁed by researchers to drive Industry 5.0. Five major themes of Industry 5.0
addressing, supply chain evaluation and optimization, enterprise innovation and digitization, smart
and sustainable manufacturing, transformation driven by IoT, AI, and Big Data, and Human-machine
connectivity were classiﬁed among the published literature, highlighting the research themes that
can be further explored. It is observed that the theme of Industry 5.0 as a gateway towards human
machine connectivity and co-existence is gaining more interest among the research community in the
recent years.
Keywords: Industry 5.0; artiﬁcial intelligence; smart manufacturing; big data; internet of things;
human-machine coexistence
1. Introduction
Today’s manufacturing industry is currently experiencing a rapid transformation
due to the onset of fast-growing digital technologies and Artiﬁcial Intelligence (AI)-based
solutions. Manufacturers throughout the world are faced with the challenge of increasing
productivity while keeping humans in loop at manufacturing industries. This task becomes
even more difﬁcult as robots become more crucial to the manufacturing process by means
of emerging technologies such as brain-machine interfaces and advances in AI. These
challenges can be addressed by the next industrial revolution, known as Industry 5.0. In
short, the concept of Industry 5.0 refers to humans and robots working as collaborators
Appl. Syst. Innov. 2022, 5, 27. https://doi.org/10.3390/asi5010027
https://www.mdpi.com/journal/asi

Appl. Syst. Innov. 2022, 5, 27
2 of 14
rather than competitors [1]. This follows the previous revolutions Industry 1.0, Industry
2.0, Industry 3.0, and Industry 4.0.
Industry 1.0 came about in the 18th century and focused on the sectors of textiles,
steam power, iron, tools, cement, chemicals, gas, lighting, glass, paper, mining, agriculture,
and transportation. The achievements of this revolution include employability, agriculture
development, transportation, and sustained growth. The noted drawbacks to Industry 1.0
include pollution and the time needed to implement the associated methodologies. Indus-
try 1.0 utilized the mathematical tools of linear programming and geometry [2]. Industry
2.0 started in the 19th century and focused on iron, steel, rail, electriﬁcation, machine tools,
paper, petroleum, chemical, maritime technology, rubber, bicycles, automobiles, applied sci-
ence, fertilizer, engines, turbines, telecommunications, and modern business management.
The achievements of this revolution include the emergence of the electrical power grid,
telephones, telegraph, and internal combustion engines. The primary drawback of Industry
2.0 is the high cost to consume electrical power. Industry 2.0 utilized the mathematical tools
of differential equations, linear equations, and geometry [2]. Industry 3.0 started in the
20th century and focused on the semiconductor industry, digital circuits, programmable
integrated circuits, telecommunication, wireless communication, the renewable energy
sector, and automation [2]. The achievements of this revolution include telecommunication,
renewable energy, automated industries, and robots. The primary drawback to Indus-
try 3.0 is that automated system would not work in certain situations [2]. For example,
one of the primary aspects of Industry 3.0 involved implementing Flexible manufacturing
Systems. (FMS). However, these systems are very complex and added extra operational
costs that were not feasible for some organizations. The complexity and added costs de-
terred many companies [3]. Industry 3.0 utilized the mathematical tools of differential
equations, linear programming, and logical controllers. Industry 4.0 came about in the 21st
century and focused on all types of industries with intelligent systems. The achievements
of this revolution include fully automated systems, artiﬁcial intelligent systems that work
in uncertain situations, with machine learning having a positive inﬂuence on the fourth
industrial revolution. The drawbacks of Industry 4.0 are that all the data in the cloud may
not be protectable, and fully expert systems are not yet developed for industries. The
mathematical tools utilized by Industry 4.0 include optimization techniques and network
theory [2].
The term “Industry 5.0” was coined by Michael Rada [4]. One of the key aspects
which Industry 5.0 entails is the use of collaborative robots that will help mitigate risk.
These robots can notice, understand, and feel the human operator as well as the goals and
expectations for the tasks being performed. The intention is that these robots will watch
and learn how an individual performs a task and help human operators in performing
the task. Furthermore, Industry 5.0 entails the penetration of AI into human life with the
aim of enhancing the man capacity. In Industry 5.0 advanced IT technologies, IoT, robots,
AI, and augmented reality are actively used in the industry for beneﬁt and convenience of
human workers [5].
Industry 5.0 recognizes the capacity of industry to fulﬁll social objectives beyond
employment and development, to become a sustainable source of development, by making
production regard the limitations of our planet and prioritizing employee health ﬁrst. To
be a trusted system for individuals seeking a satisfying and healthy career, Industry 5.0
contributes to the technology upgrade required by industry. It prioritizes worker welfare
and employs new technology to create wealth beyond employment and growth while
respecting the planet’s constraints. It empowers workers and meets their changing skill
and training requirements. It boosts industry’s competitiveness and attracts top personnel.
“An economy that works for people”, “European Green Deal” and “Europe ﬁt for the
digital age” are three goals of the Commission in implementing Industry 5.0. Therefore,
Industry 5.0 is not founded on technology but on principles such as human-centricity,
environmental stewardship, and social beneﬁt. This reorientation is grounded in the notion

Appl. Syst. Innov. 2022, 5, 27
3 of 14
that technology may be tailored to encourage values, and that technological innovation can
be built on ethical objectives, not the other way around [6].
The overall current state of understanding of Industry 5.0 describes it as the movement
to bring the human touch back to the manufacturing industry. This is driven by the
consumer’s desire for mass personalization. This understanding means that Industry 5.0
products provide consumers with a way of realizing their urge to express themselves,
and they will pay a premium to do so [7]. To summarize, Industry 5.0 is a concept that
seeks to make industry more sustainable, human centric, and resilient. Some view it as
an evolutionary, incremental advancement that builds on the concepts and practices of
industry 4.0 and others view Industry 5.0 as a complement to the Industry 4.0 paradigm.
Table 1 contrasts objectives, systemic approaches, human factors, enabling technologies
and concepts, and environmental consideration of Industry 4.0 and 5.0 [7,8]. Since Industry
5.0 is a new concept there is little agreement on how it is deﬁned. However, it is observed
that the primary trend of Industry 5.0 is the introduction of human-robot co-working
environment and the creation of smart society.
Table 1. Contrast of Industry 4.0 and Industry 5.0.
Industry 4.0
Industry 5.0
Objective
•
Smart manufacturing (smart mass
production, smart products, smart working,
smart supply-chain),
•
System(s) optimization.
•
Sustainability,
•
Environmental stewardship,
•
Human-Centricity,
•
Social beneﬁt.
Systemic Approaches
•
Real-time data monitoring,
•
Integrated chain that follows through end of
life-cycle phases.
•
Utilization of technology ethically to
advance human values and needs,
•
Socio-centric technological decisions,
•
6R methodology and logistics efﬁciency
design principles.
Human Factors
•
Human Reliability,
•
Human-computer interaction,
•
Repetitive movements.
•
Employee safety and management,
•
Learning/training for employees.
Enabling Technologies and
Concepts
•
Cloud Computing,
•
Internet of Things,
•
Big Data and Analytics,
•
Cyber Security,
•
Digitization (simulation, digital twins,
artiﬁcial intelligence, augmented, virtual, or
mixed technology),
•
Automation (advanced robotics, remote
monitoring, autonomous robots,
machine-to-machine communication),
•
Cyber-physical systems,
•
Horizontal and Vertical Integration (PLC,
Supervisory Control and Data Acquisition
(SCADA), Manufacturing Execution System
(MES), Enterprise Resource Planning (ERP)),
•
Additive Manufacturing.
•
Cloud Computing,
•
Internet Of Things,
•
Big Data and Analytics,
•
Cyber Security,
•
Digitization (simulation, digital twins,
artiﬁcial intelligence, augmented, virtual, or
mixed technology),
•
Human-machine-interaction,
•
Multi-lingual speech and gesture
recognition,
•
Tracking technologies for mental and
physical occupational strain,
•
Collaborative Robots,
•
Bio-Inspired safety and support Equipment,
•
Decision support systems,
•
Smart Grids,
•
Predictive maintenance.
Environmental
Implications
•
Systems are economic,
•
Waste prevention per data analytics, additive
manufacturing, and optimized systems,
•
Increased material consumption,
•
Increased energy usage,
•
Extended product life cycle.
•
Waste prevention and recycling,
•
Renewable Energy sources,
•
Energy-efﬁcient data storage, transmission,
and analysis,
•
Smart and energy-autonomous sensors.

Appl. Syst. Innov. 2022, 5, 27
4 of 14
To understand the perspective of what Industry 5.0 is, its evolution, and the technolo-
gies and domains that enable meeting Industry 5.0, in this paper text mining tools and
techniques are used to explore the published literature landscape to identify commonalities
and identify future directions of research in spearheading the transformation towards
Industry 5.0.
The remainder of the paper is organized as follows, Section 2 details on the data
gathering process detailing on the databases from which the data is extracted and analyzed,
Section 3 expands on the text mining approach used and highlights the ﬁndings on the
current state of Industry 5.0 research, and Section 4 concludes the article and identiﬁed
the contributions.
2. Data Gathering and Preprocessing
The data gathering process involved identifying and extracting data of published
research articles from scholarly databases. The databases used to extract data for this
study include IEEE (Institute of Electrical and Electronic Engineers), Science Direct, and
MDPI. These databases have a wider variety of coverage in terms of sources than other
databases which is important since publications focusing on Industry 5.0 may be outside the
“top” journals in the ﬁeld [9]. Further, the restricted access to the authors of the databases
available is another factor in choosing the identiﬁed sources and restricting the analysis to
the abstracts. To identify published articles addressing the topic of Industry 5.0, key term
“Industry 5.0” was used to search the metadata and identify published articles.
This included any published articles mentioning “Industry 5.0” in the title, abstract, or
the keywords. This is because Industry 5.0 is still an emerging term, and it is still unclear
what other key terms and synonyms are used [9]. The keyword search produced a total of
196 documents which included 26 from IEEE, 76 from MDPI, and 94 from Science Direct.
The time range for these documents is 2016–2022 indicating the earliest publication on
Industry 5.0 in 2016. The data collected included the publisher, title, publication year, and
abstract for each publication retrieved. The data was then sorted by the database it was
retrieved from and converted for further analysis into an .xlsx ﬁle. Once sorted, the data
was labeled to identify the publisher, title, and the abstract corresponding to each published
article retrieved. To analyze the gathered data, “R” a statistical language widely used by
statisticians and data miners was utilized to transform, visualize, and analyze the data.
This included data preprocessing, transformation, key term extraction, frequency analysis,
and topic modeling. Table 2 illustrates the search results of each individual data base the
abstracts were extracted from.
Table 2. Key Term ‘Industry 5.0’ Search Results.
Database
Abstracts
Timeline
IEEE Explore
26
2019–2022
Science Direct
94
2016–2021
MDPI
76
2018–2021
Total
196
The data is converted into a. csv format to be imported to R and then cleaned for
further analysis by removing unwanted characters such as white spaces, numbers, symbols,
and tag from the abstracts. Once these characters are removed the next steps were to delete
stop words and convert words to lower case. Several text mining tasks, facilitated by
Quanteda package [10] and dpylr package in R, include removing common words from
documents. Stop word removal is a process of removing commonly occurring words for
conjunction and propositions such as, a, I, in, for, with, the, not, on, and several similar other
words that do not usually contribute much to the meaning of a given sentence. Most text
written in English language follows punctuation and use of lower case and upper-case
text. Though capitalizations enable humans to differentiate between nouns and proper
nouns, in text analysis, words irrespective of where they are capitalized are treated equally

Appl. Syst. Innov. 2022, 5, 27
5 of 14
and thereby converting all the characters to either lower case or upper case. There is also
the process of stemming and creating vectors. Stemming helps to standardize the text
by preﬁxes, sufﬁxes, and inappropriate pluralization’s in the text document. Creating
vectors involves transforming the data into a representation to act as a suitable input for
text mining algorithms.
3. Data Analysis and Discussion
Text mining, which is also referred to intelligent text analytics, text data mining,
and text knowledge discovery, is deﬁned as the discovery of either new or previously
unknown information through the extraction of information from various written resources.
Text mining help to uncover new information and knowledge by identifying patterns in
documents from several sources [11]. Text mining may involve several other methods
such as Natural Language Progression (NLP), Information retrieval, Clustering, Document
Classiﬁcation, Web mining, Information Extraction, and Concept Extraction [12]. It has been
widely recognized among researchers that text mining’s feasibility for exploring published
literature and discovering concepts and trends across a given domain has seen tremendous
growth. For example, Bach et al. discuss the advantage of using text mining in the ﬁnancial
sector for stock market predictions [13], Aureli portrays the applicability of text mining
for studying organizations’ social and environmental reports [14], Namugera et al. use
text mining to study the social media usage of traditional media houses in Uganda to
understand the topics these media houses discuss and determine if they are positively
or negatively correlated [15], and use of text mining tools and techniques to analyze the
landscape of Model based Systems Engineering [16].
In this paper the text mining framework as illustrated in Figure 1 [17] is used for
exploring and analyzing the published articles on Industry 5.0 to identify key terms often
used and the themes into which Industry 5.0 research can be classiﬁed into using the text
data extracted.
Figure 1. Framework used for the analysis of Industry 5.0 abstracts.
3.1. Frequently Used Terms Extraction from the Data
Term extraction and frequency analysis is focused on pinpointing the relevant terms
in each collection of text. Identifying the unique terms with statistical techniques such
as calculating relative term frequency among the documents in a dataset enables better
understanding of the information provided from the text [18]. The Inverse Documentation
Frequency (IDF) measure is a widely utilized method for determining the contribution of
terms. The greatest advantage of IDF is that it can aid in determining the inﬂuence of term
in a group of given documents by identifying the frequency of term in a document and
the number of times it occurs in a document [18,19]. When it comes to “low” and “high”

Appl. Syst. Innov. 2022, 5, 27
6 of 14
IDF values, low values represent less informative terms appearing in several documents,
and high values represent more informative terms appearing in only a few documents. In
this paper, frequency analysis was utilized to identify the primary terms associated with
Industry 5.0. Table 3 represents the top 20 frequently observed terms in the data along with
their corresponding frequency.
Table 3. Top 20 terms identiﬁed from the database of Industry 5.0.
Terms Usage Identiﬁed
Term Frequency
industrial revolution
45
artiﬁcial intelligence
43
supply chain
32
big data
28
digital transformation
24
machine learn
23
industry technology
22
digital twin
19
recent year
18
cloud compute
18
thing iot
18
sustainable development
17
future research
17
intelligence ai
16
smart manufacture
16
digital technology
16
fourth industrial
16
manufacture industry
16
production system
15
manufacture system
15
The term “Industrial Revolution” was observed to occur the most in the database.
This is indeed expected since Industry 5.0 is referred quite often to as the ﬁfth industrial
revolution. Several publications in the data gathered use this term in comparing the dif-
ferent industrial revolutions throughout the years. Industry 5.0, just as past industrial
revolutions, is predicted to have major impacts on the dynamics of socio-economic systems
more speciﬁcally to have a large impact in industrial production systems [20]. The term
“Artiﬁcial Intelligence” is observed to have the second highest count. Artiﬁcial intelli-
gence seems to be one of the central components of Industry 5.0, mostly addressed for
automating manufacturing processes, furthering the primary focus of cooperation between
man and machine [21]. The term with the third highest count was “Supply Chain.” This
is highly signiﬁcant since it is believed that Industry 5.0 will inﬂuence supply chains to
an unprecedented level. The trends for Industry 5.0 supply chains include the incorporation
of collaborative robots (co-bots), intelligent systems, mass personalization, and mass cus-
tomization [22]. The term with the fourth highest count is “Big Data.” Big Data is integral to
Industry 5.0. It is believed that Industry 5.0 will introduce new innovations in management
framework that takes Big Data into consideration. Furthermore, Big Data will be crucial
for reaping the maximum beneﬁt of Industry 5.0 such as modern technologies and new
innovations in Internet of Things (IoT) and artiﬁcial intelligence [23]. The term “Digital
Transformation” has the ﬁfth highest count. This follows the prediction that Industry 5.0
will bring about a transformation towards digital platforms and a digital economy. In all,
it is predicted that there will be a digital ecosystem, an open, distributed, self-organizing,
system of system. The intention of this digital transformation is to unite subsystems to
provide a common information space with access to a rich set of re-usable applied services
that can support resource planning and control in real time. The digital transformation for
Industry 5.0 should also provide standard access to cloud resources and services and to data
perceived by external smart sensor networks [24]. The use of terms “manufacture industry”,
“production system”, and “manufacture system” are centered on the dialogue from the

Appl. Syst. Innov. 2022, 5, 27
7 of 14
research community discussing the shift from Industry 4.0 to 5.0 in the manufacturing and
production engineering domain for the plausibility of advanced human machine interfaces
for improved integration and better automation.
3.2. Term Frequency Analysis
The terms identiﬁed from the overall data set enabled to gain understanding on what
speciﬁc terms were more focused upon by the researchers addressing Industry 5.0 paradigm.
A measure of frequency for the identiﬁed terms is used to plot on a line graph, the relative
use of terms over the years to identify their usage trends by the researchers. Based on the
terms extracted in Section 3.2, the following terms were identiﬁed for analysis (a) “twin”—
to understand the trend on exploring the use of digital twins in enabling Industry 5.0,
(b) “data”—for exploring the trend on use of big data i.e., set of diverse actionable data to
empower Industry 5.0, (c) “intelligence”—to understand the trend on the use of artiﬁcial
intelligence to aid Industry 5.0, (d) “cloud”—for understanding the trend in exploring
the use of cloud based technologies and cloud computing as an enabler of Industry 5.0,
(e) “IoT”—for exploring the trend on identifying IoT as an enabler for Industry 5.0, and
(f) “machine”—for exploring the dialogue on the use of machine learning towards Industry
5.0 transformation. Figure 2 illustrates the trends on the use of the aforementioned terms
over the past years. The use of the terms starting from year 2016 in the graph indicates
to the limitation of the data that was gathered for analysis, starting from year 2016 where
the ﬁrst peer reviewed publication on Industry 5.0 was observed. Please see Table 1. The
use of term “twin” in 2016 indicates to the initial attempt at exploring the use of digital
twins to address Industry 5.0. Starting year 2018 the use of terms “intelligence” “cloud”,
and “IoT” are observed, indicating an initial interest in exploring artiﬁcial intelligence, IoT,
and cloud computing technologies as enabler of Industry 5.0, with more interest in IoT.
A peak in use of actionable data sets i.e., Big Data in the year 2021 followed by IoT and
machine learning indicate more interest among the research communities to explore from
an integration perspective toward a well-connected, distributed, intelligent, and actionable
human centric systems. Please note that the abrupt drop in the term usage reﬂects to the
limitation on the data gathered on publications until early 2022 considered for the analysis.
Figure 2. Terms usage trends in the context of Industry 5.0.
3.3. Topic Analysis
Topic Analysis was used to understand the major themes around which the published
literature on Industry 5.0 can be classiﬁed. Topic analysis from text mining is deﬁned as the
act of extracting topics or thematic elements from a given set of documents. Topic analysis

Appl. Syst. Innov. 2022, 5, 27
8 of 14
focuses on the characterization of a topic based on distribution of terms and the mixture of
each topic in a document [25]. One of the most common methods of topic analysis from
text mining is “Latent Dirichlet Allocation” (LDA) [26]. LDA is an unsupervised machine
learning method mostly used for applications such as opinion modeling, extracting topics
from source codes, and hashtag recommendations [17]. One of the greatest advantages of
LDA is its pertinence to several domains while taking three domains into consideration,
documents, words, and topics. LDA enables the user to deﬁne the number of topics as
a parameter that the textural data must be characterized into. It is to be noted that, if this
parameter is small, the topic identiﬁed provide only few semantic contexts whereas when
the parameter to too large there is a scope for the topics identiﬁed to overlap. This, after
several trails the authors limit the number of topics the data is to be characterized into 5 to
be more reﬂective and coherent to Industry 5.0. Table 4 depicts the top ﬁve topics, and ten
most likely terms observed in each topic, and a representative label for each topic provided
by the authors.
Table 4. Topics and Topic Labels identiﬁed in context of published Industry 5.0 articles.
Topic
Number
Topic Terms
Topic Label in Context of
Industry 5.0
Topic 1
supply, approach, engineer, model, result, safety,
analysis, method, performance, key
Supply Chain Evaluation and
Optimization
Topic 2
digital, research, study, industry, innovation,
company, review, business, organization, future
Enterprise Management,
Innovation, and Digitization
Topic 3
manufacture, smart, industry, revolution,
sustainable, technology, industrial, energy,
technological, challenge
Smart and Sustainable
Manufacturing
Topic 4
IoT, security, internet, datum, thing, compute,
system, device, health, cloud
Transformation driven by IoT,
Bigdata, and AI
Topic 5
human, robot, system, production, process, task,
industry, intelligent, robotic, manufacture
Human Machine connectivity
and co-existence
Figure 3 represents the topic distribution across the entire dataset of 196 abstracts
on Industry 5.0 and Figure 4 portrays the spread of the identiﬁed topics among the data
gathered over the period of years 2016 to 2022.
Figure 3. Topic distribution in the data extracted.

Appl. Syst. Innov. 2022, 5, 27
9 of 14
Figure 4. Industry 5.0 Themes and their respective spread over time.
The top abstracts, in a descending order, that represent each individual topics (Table 4)
have been summarized to explore and better understand the perspective in the context of
Industry 5.0 being addressed so far through published articles
Theme 1—Industry 5.0 in context of supply chain evaluation and optimization in manufactur-
ing processes: This topic supports the exploration of how Industry 5.0 can enable supply
chain evaluation and optimization in manufacturing processes. More speciﬁcally, use of
a multi-objective mathematical model to design a sustainable-resilient supply chain based
on strategic and tactical decision levels [27], optimization of mining methods using multi-
level, multi-factor, multi-objective, and multi-index comprehensive evaluation system
involving technology, economy, and safety [28], exploring Social Value Orientation theory
for understanding decision making preferences for join resource allocation [29], exploring
the constructs of Industry 5.0 for supporting supply chain operations [22], research direc-
tions for supply chain transformations [30], inﬂuence of industrial internet of things and
emerging technologies on digital transformation capabilities of organizations [31], effective-
ness indicators for enterprise resource planning systems to aid digitization of information
ﬂows [32], enabling constructs of Industry 5.0 to control and manage supply chains in
emergency [33], future of supply chains in context of Industry 5.0 [34], and approaches for
supply chain digitization [35]. Addressing the 13% of the data gathered, a constant interest
among the researchers is observed over time on this topic. Figure 4 portrays this trend.
Theme 2—Industry 5.0 in context of enterprise management, innovation, and digitization:
Composed of 27% of the data analyzed, this thematic topic addresses the construct of
Industry 5.0 in context of Enterprise Management, Innovation, and Digitization. This
theme addresses research on translating critical success factors of project management in
relation to Industry 4.0 for sustainability in manufacturing enterprise [36], an absolute
innovation management framework for addressing the importance making innovation
more understandable, implementable, and part of routine in organizations [23] required
for adopting to the constructs of Industry 5.0, importance of addressing the nexus of
entrepreneurial leadership and product innovation through design thinking [37] a core
need for moving towards the notion of Industry 5.0 in organizations, identiﬁcation of
how digital product and process innovations might affect proﬁtable customer strategies in
a global context [38], use of biological resources and policy to drive Industry 5.0 [39], using

Appl. Syst. Innov. 2022, 5, 27
10 of 14
sustainability based metrics for digital technologies [40] to enhance production operations,
identiﬁcation of value drives for successful digital transformations [41], signiﬁcance and
adoption of global reporting initiative standards in context of technology sustainability [42],
enablers and challenges of digitization [43], and the importance of technological adoption
and development based on the needs and demands of society [44]. This thematic aspect
is observed to have a constant interest among the researchers over time for addressing
Industry 5.0. Figure 4 portrays this trend.
Theme 3—Industry 5.0 in context of smart and sustainable manufacturing: Representing
25% of the data gathered, this major thematic aspect addresses Industry 5.0 in context
of enabling smart and sustainable manufacturing. This thematic aspect is also observed
to have a constant interest among the researchers over time for addressing Industry 5.0.
Figure 4 portrays this trend. This theme majorly addresses pathways to manufacturing
systems that can adopt by exploring the drivers and barriers manufacturing systems might
face when seeking a transition to smart and sustainable paradigms towards Industry 4.0
and beyond [45], impact of industrial mathematics on industrial revolutions and how it
can enable smart industries to meet customer need for future uncertain business environ-
ments [2], incorporating sustainable manufacturing measures for opportunities towards
smart manufacturing [46], identiﬁcation of components that enable Industry 5.0 for in-
telligent production systems [47], applicability of cloud based decision making based
on information acquired from sensors [48], enabling smart manufacturing processes [49],
machine learning enabled power dispatch systems [50], digital twins for sustainable opera-
tions [51], biomimetic designs for industry 5.0 [52], and integration of software suites and
digital technologies for effective manufacturing quality management systems [53].
Theme 4—Industry 5.0 transformation driven by IoT, Bigdata, and AI: This theme address-
ing the use of IoT, big data, and AI towards Industry 5.0 is observed gain a lot of attention
recently among the research community (illustrated in Figure 4) though it encompasses
only 11% of published research articles so far (illustrated in Figure 3). This topic further
relates to utilizing IoT technology such as sensors and actuators into the industrial process
of Industry 5.0 to aid in the mass customization of products [54]. Research in theme is
addressed on, taxonomy for integration of blockchain and machine learning in an IoT
environment [55], expanding technological infrastructure, provision of budgetary support
based on sustainable business models, standardization, and synchronization protocols,
improving stakeholders’ engagement and involvement [56], taxonomy analysis to aid in
implementing methods and algorithms for different IoT application [57], in identifying
techniques to improve the security and efﬁciency of data transmission between the IoT
devices [58], exploring the use of deep learning and AI for monitoring [59], use of amazon
web services using IoT and cyber physical systems for equipment monitoring [60], chal-
lenges and impediments of IoT [61,62], and energy efﬁciency and assessment models using
big data and AI [63].
Theme 5—Human machine connectivity and co-existence: This theme relates to emergence
of Industry 5.0 as the concept of human-robot/human-machine coexistence. This refers to
the aspect of humans and robots in loop supporting and assisting each other in manufactur-
ing and production engineering processes [64]. Research addressed in this theme include
knowledge-based tasks and automation for humans and robots to measure cycle times [65],
exploring the application of social value orientation theory to human machine contexts and
multiagent systems [29], scientiﬁc improvements transforming the production lines and
machines in intelligent systems [47], soft robotics for industrial applications involving ma-
nipulation of fragile objects [66], achieving a balance between capital and labor welfare by
deploying Industry 4.0 technologies with a worker centric approach [67], and use of AI for
robust solutions in mobile robotics [68]. Further, research in this theme also addresses the
need for resilient workforce for adapting to workplaces and enterprises [69], perspectives
on human centricity in future smart manufacturing [70], and use of agent-based approach
to explore effects of human-robot interactions [71].

Appl. Syst. Innov. 2022, 5, 27
11 of 14
4. Conclusions
Enabled by the capabilities of text mining techniques, in this paper an attempt to
understand and classify Industry 5.0 based on the published research articles with the time
frame of when the term was ﬁrst coined i.e., 2016 to the year 2022 is portrayed. Addressing
the objective of the paper, term extraction technique was used to identify the most often
occurring terms in the abstract text data gathered on Industry 5.0 related publications.
The terms artiﬁcial intelligence, big data, supply chain, digital transformation, machine
learning were observed to be most referred to. This coincides with the fact that Industry
5.0 is seen to facilitate repetitive tasks with the use of artiﬁcial intelligence and machine
learning technologies, parallelly assisting humans for cognitive support. Further, big data
and digital transformation are foreseen to provide an information space rich with data
that can be used for resource planning and control in real time. Topic analysis technique
was used to identify the thematic aspects of papers published addressing Industry 5.0.
Five different thematic aspects were observed across the landscape, with most of them in the
context of smart and sustainable manufacturing followed by human machine connectivity
and co-existence. More speciﬁcally, the theme of Industry 5.0 as a gateway towards
human machine connectivity and co-existence is observed to gain a lot of interest among
the research community. Further a brief description of the top ten papers from the data
deﬁning the topics is provided for the audience to understand the perspective and relevance
of topics.
By examining the analysis provided future research directions can be identiﬁed and
predicted on how Industry 5.0 will impact the manufacturing landscape in the years to
come. The results obtained are limited to the data i.e., 196 abstracts extracted. It is to be
noted that these the results are subject to change with increase in the number of abstracts
used for analysis along with expanding the digital libraries used for extracting the data.
As a future work, it is believed a comprehensive analysis along with integration of data
crawling techniques will provide a better perspective on what Industry 5.0 is and how it is
perceived among the research community.
Author Contributions: Conceptualization, A.A. and S.L.; methodology, A.A.; formal analysis, A.A.,
W.A. and S.L.; investigation, A.A., A.L. and I.E.; resources, A.A.; data curation, A.A., D.E. and W.A.;
writing—original draft preparation, A.A. and D.E.; writing—review and editing, A.A. and A.L;
visualization, A.A. All authors have read and agreed to the published version of the manuscript.
Funding: This research received no external funding.
Institutional Review Board Statement: Not applicable.
Informed Consent Statement: Not applicable.
Data Availability Statement: The data presented in this study are available on request from the
corresponding author.
Conﬂicts of Interest: The authors declare no conﬂict of interest.
References
1.
Nahavandi, S. Industry 5.0—A human-centric solution. Sustainability 2019, 11, 4371. [CrossRef]
2.
Vinitha, K.; Prabhu, R.A.; Bhaskar, R.; Hariharan, R. Review on industrial mathematics and materials at Industry 1.0 to Industry
4.0. Mater. Today Proc. 2020, 33, 3956–3960. [CrossRef]
3.
Madsen, E.S.; Bilberg, A.; Hansen, D.G. Industry 4.0 and digitalization call for vocational skills, applied industrial engineering,
and less for pure academics. In Proceedings of the 5th P&OM World Conference, Production and Operations Management,
P&OM, Havana, Cuba, 6–10 September 2016.
4.
Rada, M. Industry 5.0-from Virtual to Physical. LinkedIn. 7 March 2018. Available online: https://www.linkedin.com/pulse/
industry-50-from-virtual-physical-michael-rada (accessed on 3 February 2022).
5.
Skobelev, P.O.; Borovik, S.Y. On the way from Industry 4.0 to Industry 5.0: From digital manufacturing to digital society. Industry
4.0 2017, 2, 307–311.
6.
Müller, J. Enabling Technologies for Industry 5.0: Results of a Workshop with Europe’s Technology Leaders; European Commission:
Brussels, Belgium, 2020.

Appl. Syst. Innov. 2022, 5, 27
12 of 14
7.
Østergaard, E.H. Welcome to Industry 5.0. Available online: https://info.universal-robots.com/hubfs/Enablers/White%20
papers/Welcome%20to%20Industry%205.0_Esben%20%C3%98stergaard.pdf?submissionGuid=00c4d11f-80f2-4683-a12a-e821
221793e3 (accessed on 3 February 2022).
8.
Saurabh, S.; Ambad, P.; Bhosle, S. Industry 4.0–A glimpse. Procedia Manuf. 2018, 20, 233–238.
9.
Madsen Øivind, D.; Berg, T. An Exploratory Bibliometric Analysis of the Birth and Emergence of Industry 5. Appl. Syst. Innov.
2021, 4, 87. [CrossRef]
10.
Welbers, K.; Van Atteveldt, W.; Benoit, K. Text Analysis in R. Commun. Methods Meas. 2017, 11, 245–265. [CrossRef]
11.
Hearst, M. What is text mining. SpringerPlus 2016, 5, 1608.
12.
Feng, L.; Chiam, Y.K.; Lo, S.K. Text-Mining Techniques and Tools for Systematic Literature Reviews: A Systematic Literature
Review. In Proceedings of the 2017 24th Asia-Paciﬁc Software Engineering Conference (APSEC), Nanjing, China, 4–8 December
2017; pp. 41–50.
13.
Bach, M.P.; Krsti´c, Ž.; Seljan, S.; Turulja, L. Text Mining for Big Data Analysis in Financial Sector: A Literature Review. Sustainability
2019, 11, 1277. [CrossRef]
14.
Aureli, S. A comparison of content analysis usage and text mining in CSR corporate disclosure. Int. J. Digit. Account. Res. 2017,
17, 1–32. [CrossRef]
15.
Namugera, F.; Wesonga, R.; Jehopio, P. Text mining and determinants of sentiments: Twitter social media usage by tradi-tional
media houses in Uganda. Comput. Soc. Netw. 2019, 6, 3. [CrossRef]
16.
Akundi, A.; Mondragon, O. Model based systems engineering—A text mining based structured comprehensive overview. Syst.
Eng. 2021, 25, 51–67. [CrossRef]
17.
Wiedemann, G.; Niekler, A. Hands-On: A Five Day Text Mining Course for Humanists and Social Scientists in R. Available online:
http://ceur-ws.org/Vol-1918/wiedemann.pdf (accessed on 10 November 2021).
18.
Christian, H.; Pramodana, A.M.; Suhartono, D. Single document automatic text summarization using term frequency-inverse
document frequency (TF-IDF). ComTech Comput. Math. Eng. Appl. 2016, 7, 285–294. [CrossRef]
19.
Neto, J.L.; Santos, A.D.; Kaestner, C.A.A.; Alexandre, N.; Santos, D.; Celso, A.A.; Alex, K.; Freitas, A.A.; Parana, C. Document
Clustering and Text Summarization. 2000. Available online: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.43.4634
(accessed on 10 November 2021).
20.
Melnyk, L.; Kubatko, O.; Dehtyarova, I.; Matsenko, O.; Rozhko, O. The effect of industrial revolutions on the transformation of
social and economic systems. Probl. Perspect. Manag. 2019, 17, 381–391. [CrossRef]
21.
Haleem, A.; Javaid, M. Industry 5.0 and its applications in orthopedics. J. Clin. Orthop. Trauma 2019, 10, 807. [CrossRef]
22.
Frederico, G.F. From supply chain 4.0 to supply chain 5.0: Findings from a systematic literature review and research direc-tions.
Logistics 2021, 5, 49. [CrossRef]
23.
Aslam, F.; Aimin, W.; Li, M.; Rehman, K.U. Innovation in the Era of IoT and Industry 5.0: Absolute Innovation Management
(AIM) Framework. Information 2020, 11, 124. [CrossRef]
24.
Gorodetsky, V.; Larukchin, V.; Skobelev, P. Conceptual Model of Digital Platform for Enterprises of Industry 5.0. In Proceedings of
the Foundations of Computational Intelligence; Springer Science and Business Media: Saint-Petersburg, Russia, 2020; Volume 3,
pp. 35–40.
25.
Liu, L.; Tang, L.; Dong, W.; Yao, S.; Zhou, W. An overview of topic modeling and its current applications in bioinformatics.
SpringerPlus 2016, 5, 1608. [CrossRef]
26.
Jelodar, H.; Wang, Y.; Yuan, C.; Feng, X.; Jiang, X.; Li, Y.; Zhao, L. Latent Dirichlet allocation (LDA) and topic modeling: Models,
applications, a survey. Multimed. Tools Appl. 2019, 78, 15169–15211. [CrossRef]
27.
Sazvar, Z.; Tafakkori, K.; Oladzad, N.; Nayeri, S. A capacity planning approach for sustainable-resilient supply chain network
design under uncertainty: A case study of vaccine supply chain. Comput. Ind. Eng. 2021, 159, 107406. [CrossRef]
28.
Guo, Q.; Yu, H.; Dan, Z.; Li, S. Mining Method Optimization of Gently Inclined and Soft Broken Complex Ore Body Based on
AHP and TOPSIS: Taking Miao-Ling Gold Mine of China as an Example. Sustainability 2021, 13, 12503. [CrossRef]
29.
Mizrahi, D.; Zuckerman, I.; Laufer, I. Using a Stochastic Agent Model to Optimize Performance in Divergent Interest Tacit
Coordination Games. Sensors 2020, 20, 7026. [CrossRef] [PubMed]
30.
Barata, J. The fourth industrial revolution of supply chains: A tertiary study. J. Eng. Technol. Manag. 2021, 60, 101624. [CrossRef]
31.
Ghosh, S.; Hughes, M.; Hodgkinson, I.; Hughes, P. Digital transformation of industrial businesses: A dynamic capability approach.
Technovation 2021, 102414. [CrossRef]
32.
Anguelov, K. Indicators for the Effectiveness and Efﬁciency of the Implementation of an Enterprise Resource Planning System. In
Proceedings of the 2021 12th National Conference with International Participation (ELECTRONICA), Soﬁa, Bulgaria, 27–28 May
2021; pp. 1–4.
33.
Khurana, S.; Haleem, A.; Luthra, S.; Huisingh, D.; Mannan, B. Now is the time to press the reset button: Helping India’s
companies to become more resilient and effective in overcoming the impacts of COVID-19, climate changes and other crises. J.
Clean. Prod. 2021, 280, 124466. [CrossRef] [PubMed]
34.
Maddikunta, P.K.R.; Pham, Q.-V.; Prabadevi, B.; Deepa, N.; Dev, K.; Gadekallu, T.R.; Ruby, R.; Liyanage, M. Industry 5.0: A survey
on enabling technologies and potential applications. J. Ind. Inf. Integr. 2021, 100257. [CrossRef]
35.
Deepu, T.S.; Ravi, V. A conceptual framework for supply chain digitalization using integrated systems model approach and
DIKW hierarchy. Intell. Syst. Appl. 2021, 10–11, 200048. [CrossRef]

Appl. Syst. Innov. 2022, 5, 27
13 of 14
36.
Vrchota, J.; ˇRehoˇr, P.; Maˇríková, M.; Pech, M. Critical Success Factors of the Project Management in Relation to Industry 4.0 for
Sustainability of Projects. Sustainability 2020, 13, 281. [CrossRef]
37.
Rehman, K.; Aslam, F.; Mata, M.; Martins, J.; Abreu, A.; Lourenço, A.M.; Mariam, S. Impact of Entrepreneurial Leadership
on Product Innovation Performance: Intervening Effect of Absorptive Capacity, Intra-Firm Networks, and Design Thinking.
Sustainability 2021, 13, 7054. [CrossRef]
38.
Petersen, J.A.; Paulich, J.B.; Khodakarami, F.; Spyropoulou, S.; Kumar, V. Customer-based Execution Strategy in a Global Digital
Economy. Int. J. Res. Mark. 2021, in press. [CrossRef]
39.
Schütte, G. What kind of innovation policy does the bioeconomy need? New Biotechnol. 2018, 40, 82–86. [CrossRef]
40.
Cricelli, L.; Strazzullo, S. The Economic Aspect of Digital Sustainability: A Systematic Review. Sustainability 2021, 13, 8241.
[CrossRef]
41.
Ghobakhloo, M.; Fathi, M.; Iranmanesh, M.; Maroufkhani, P.; Morales, M.E. Industry 4.0 ten years on: A bibliometric and
systematic review of concepts, sustainability value drivers, and success determinants. J. Clean. Prod. 2021, 302, 127052. [CrossRef]
42.
Narula, S.; Puppala, H.; Kumar, A.; Frederico, G.F.; Dwivedy, M.; Prakash, S.; Talwar, V. Applicability of industry 4.0 technologies
in the adoption of global reporting initiative standards for achieving sustainability. J. Clean. Prod. 2021, 305, 127141. [CrossRef]
43.
Kalsoom, T.; Ahmed, S.; Raﬁ-Ul-Shan, P.M.; Azmat, M.; Akhtar, P.; Pervez, Z.; Imran, M.A.; Ur-Rehman, M. Impact of IoT on
Manufacturing Industry 4.0: A New Triangular Systematic Review. Sustainability 2021, 13, 12506. [CrossRef]
44.
Potocan, V. Technology and Corporate Social Responsibility. Sustainability 2021, 13, 8658. [CrossRef]
45.
Bellandi, M.; De Propris, L. Local Productive Systems’ Transitions to Industry 4.0+. Sustainability 2021, 13, 13052. [CrossRef]
46.
Abubakr, M.; Abbas, A.T.; Tomaz, I.; Soliman, M.S.; Luqman, M.; Hegab, H. Sustainable and smart manufacturing: An in-tegrated
approach. Sustainability 2020, 12, 2280. [CrossRef]
47.
Massaro, A. Information Technology Infrastructures Supporting Industry 5.0 Facilities. In Electronics in Advanced Research
Industries; Wiley-IEEE Press: Hoboken, NJ, USA, 2021; pp. 51–101.
48.
Coito, T.; Firme, B.; Martins, M.; Vieira, S.; Figueiredo, J.; Sousa, J. Intelligent Sensors for Real-Time Decision-Making. Automation
2021, 2, 62–82. [CrossRef]
49.
Shariati, M.; Weber, W.E.; Bohlen, J.; Kurz, G.; Letzig, D.; Höche, D. Enabling intelligent Mg-sheet processing utilizing efﬁcient
machine-learning algorithm. Mater. Sci. Eng. A 2020, 794, 139846. [CrossRef]
50.
Yin, L.; Gao, Q.; Zhao, L.; Zhang, B.; Wang, T.; Li, S.; Liu, H. A review of machine learning for new generation smart dispatch in
power systems. Eng. Appl. Artif. Intell. 2020, 88, 103372. [CrossRef]
51.
Coupry, C.; Noblecourt, S.; Richard, P.; Baudry, D.; Bigaud, D. BIM-Based Digital Twin and XR Devices to Improve Maintenance
Procedures in Smart Buildings: A Literature Review. Appl. Sci. 2021, 11, 6810. [CrossRef]
52.
Du Plessis, A.; Broeckhoven, C.; Yadroitsava, I.; Yadroitsev, I.; Hands, C.H.; Kunju, R.; Bhate, D. Beautiful and functional: A review
of biomimetic design in additive manufacturing. Addit. Manuf. 2019, 27, 408–427. [CrossRef]
53.
Ammar, M.; Haleem, A.; Javaid, M.; Bahl, S.; Verma, A.S. Implementing Industry 4.0 technologies in self-healing materials and
digitally managing the quality of manufacturing. Mater. Today Proc. 2021, 51. [CrossRef]
54.
Fraga-Lamas, P.; Lopes, S.I.; Fernández-Caramés, T.M. Green IoT and Edge AI as Key Technological Enablers for a Sustainable
Digital Transition towards a Smart Circular Economy: An Industry 5. 0 Use Case. Sensors 2021, 21, 5745. [CrossRef]
55.
Miglani, A.; Kumar, N. Blockchain management and machine learning adaptation for IoT environment in 5G and beyond
networks: A systematic review. Comput. Commun. 2021, 178, 37–63. [CrossRef]
56.
Mbunge, E.; Muchemwa, B.; Jiyane, S.; Batani, J. Sensors and healthcare 5.0: Transformative shift in virtual care through emerging
digital health technologies. Glob. Health J. 2021, 5, 169–177. [CrossRef]
57.
Chegini, H.; Naha, R.K.; Mahanti, A.; Thulasiraman, P. Process Automation in an IoT–Fog–Cloud Ecosystem: A Survey and
Taxonomy. IoT 2021, 2, 6. [CrossRef]
58.
Vithanage, N.N.N.; Thanthrige, S.S.H.; Kapuge, M.C.K.P.; Malwenna, T.H.; Liyanapathirana, C.; Wijekoon, J.L. A Secure
Corroboration Protocol for Internet of Things (IoT) Devices Using MQTT Version 5 and LDAP. In Proceedings of the 2021
International Conference on Information Networking (ICOIN), Jeju Island, Korea, 13–16 January 2021; pp. 837–841.
59.
Sujith, A.; Sajja, G.S.; Mahalakshmi, V.; Nuhmani, S.; Prasanalakshmi, B. Systematic review of smart health monitoring using
deep learning and Artiﬁcial intelligence. Neurosci. Inform. 2021, 2, 100028. [CrossRef]
60.
Dineva, K.; Atanasova, T. Design of Scalable IoT Architecture Based on AWS for Smart Livestock. Animals 2021, 11, 2697.
[CrossRef]
61.
Omolara, A.E.; Alabdulatif, A.; Abiodun, O.I.; Alawida, M.; Alabdulatif, A.; Alshoura, W.H.; Arshad, H. The internet of things
security: A survey encompassing unexplored areas and new in-sights. Comput. Secur. 2022, 112, 102494. [CrossRef]
62.
Miraz, M.H.; Ali, M.; Excell, P.S.; Picking, R. Internet of Nano-Things, Things and Everything: Future Growth Trends. Future
Internet 2018, 10, 68. [CrossRef]
63.
Anthopoulos, L.; Kazantzi, V. Urban energy efﬁciency assessment models from an AI and big data per-spective: Tools for policy
makers. Sustain. Cities Soc. 2022, 76, 103492. [CrossRef]
64.
Demir, K.A.; Döven, G.; Sezen, B. Industry 5.0 and Human-Robot Co-working. Procedia Comput. Sci. 2019, 158, 688–695. [CrossRef]
65.
Fast-Berglund, Åsa; Thorvald, P. Variations in cycle-time when using knowledge-based tasks for humans and robots. IFAC-
PapersOnLine 2021, 54, 152–157. [CrossRef]

Appl. Syst. Innov. 2022, 5, 27
14 of 14
66.
Massaro, A. State of the Art and Technology Innovation. In Electronics in Advanced Research Industries; Wiley-IEEE Press: Hoboken,
NJ, USA, 2021; pp. 1–49.
67.
Margherita, E.G.; Braccini, A.M. Socio-technical perspectives in the Fourth Industrial Revolution-Analysing the three main
visions: Industry 4.0, the socially sustainable factory of Operator 4.0 and Industry 5.0. In Proceedings of the 7th International
Workshop on Socio-Technical Perspective in IS Development (STPIS 2021), Trento, Italy, 14–15 October 2021.
68.
Cebollada, S.; Payá, L.; Flores, M.; Peidró, A.; Reinoso, O. A state-of-the-art review on mobile robotics tasks using artiﬁcial
intelligence and visual data. Expert Syst. Appl. 2021, 167, 114195. [CrossRef]
69.
Romero, D.; Stahre, J. Towards the Resilient Operator 5.0: The Future of Work in Smart Resilient Manufacturing Systems. Procedia
CIRP 2021, 104, 1089–1094. [CrossRef]
70.
Wang, L. A futuristic perspective on human-centric assembly. J. Manuf. Syst. 2021, 62, 199–201. [CrossRef]
71.
Martin, L.; Gonzalez-Romo, M.; Sahnoun, M.; Bettayeb, B.; He, N.; Gao, J. Effect of Human-Robot Interaction on the Fleet Size
of AIV Transporters in FMS. In Proceedings of the 2021 1st International Conference on Cyber Management and Engineering
(CyMaEn), Hammamet, Tunisia, 26–28 May 2021; pp. 1–5.

---

## 7. Retraction: A New Machine Learning Approach Based on Fuzzy Data Correlation for Recognizing Sports Activities (IEEE Access (2023) DOI: 10.1109/ACCESS.2023.3298778)
**Authors:** N/A
**Year:** 2024  
**Citations:** 0  
**Source:** Scopus  
**DOI:** 10.1109/ACCESS.2024.3429868  
**URL:** https://api.elsevier.com/content/abstract/scopus_id/85199689216  

*No text available.*

---
