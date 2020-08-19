DATASET_LOCATION = "D:\\GitHub\\data\\gesto-subset\\"
DATASET_LOCATION = "D:\\GitHub\\data\\gesto-dataset\\"
DATASET_LOCATION = "C:\\gesto-dataset\\"

DATASET_REGEX = [
                    "subject(\d+)\\\(angry|sad|happy|neutral)\\\(\d+)\.png",
                    {"subject":1,"class":2,"id":3}
                ]

TASKS = [

        {   "name":"pca_svm",
            "description":"Apply pca feature extraction then svm",
                "preprocessing":{   "method":None,
                                    "parameters":None},

                "feature_extraction":{  "method":"PCA",
                                        "parameters":{"n_components":0.95}},
                "classification":{   "method":"SVM",
                                    "parameters":{}},
                "iterations":3      },

        {   "name":"ld_svm",
            "description":"Apply landmark distance feature extraction then SVM",
                "preprocessing":{   "method":None,
                                    "parameters":None},

                "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                        "parameters":{}},
                "classification":{   "method":"DecisionTree",
                                    "parameters":{}},
                "iterations":3      },

        {   "name":"pca_svm_rbf",
            "description":"Apply pca feature extraction then svm rbf",
                "preprocessing":{   "method":None,
                                    "parameters":None},

                "feature_extraction":{  "method":"PCA",
                                        "parameters":{"n_components":0.95}},
                "classification":{   "method":"SVM",
                                    "parameters":{'kernel':'rbf'}},
                "iterations":3      },

        {   "name":"ld_svm_rbf",
            "description":"Apply landmark distance feature extraction then SVM rbf",
                "preprocessing":{   "method":None,
                                    "parameters":None},

                "feature_extraction":{  "method":"LandmarkDistance",
                                        "parameters":{}},
                "classification":{   "method":"SVM",
                                    "parameters":{'kernel':'rbf'}},
                "iterations":3      },

    {   "name":"ld-pca_svm",
        "description":"Apply landmark distance + pca feature extraction then SVM",
            "preprocessing":{   "method":None,
                                "parameters":None},

            "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                    "parameters":{}},
            "classification":{   "method":"SVM",
                                "parameters":{}},
            "iterations":3      },
    {   "name":"ld-pca_svm_rbf",
        "description":"Apply landmark distance + pca feature extraction then SVM rbf",
            "preprocessing":{   "method":None,
                                "parameters":None},

            "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                    "parameters":{}},
            "classification":{   "method":"SVM",
                                "parameters":{'kernel':'rbf'}},
            "iterations":3      },

{   "name":"ld-pca_lda",
    "description":"Apply landmark distance + pca feature extraction then lda",
        "preprocessing":{   "method":None,
                            "parameters":None},

        "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                "parameters":{}},
        "classification":{   "method":"LDA",
                            "parameters":{}},
        "iterations":3      },

{   "name":"ld-pca_qda",
    "description":"Apply landmark distance + pca feature extraction then lda",
        "preprocessing":{   "method":None,
                            "parameters":None},

        "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                "parameters":{}},
        "classification":{   "method":"QDA",
                            "parameters":{}},
        "iterations":3      },

        {   "name":"ld-pca_qda-happy_svm-unhappy",
            "description":"""Apply the landmark distance feature extraction
                             and concatenate with PCA, then apply QDA to
                             find if happy then apply SVM if else""",
                "preprocessing":{   "method":None,
                                    "parameters":None},

                "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                        "parameters":{}},
                "classification":{   "method":"QDAplus",
                                    "parameters":{"othermodel":"SVM"}},
                "iterations":3      },

        {   "name":"ld-pca_qda-happy_dt-unhappy",
            "description":"""Apply the landmark distance feature extraction
                             and concatenate with PCA, then apply QDA to
                             find if happy then apply DecisionTree if else""",
                "preprocessing":{   "method":None,
                                    "parameters":None},

                "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                        "parameters":{}},
                "classification":{   "method":"QDAplus",
                                    "parameters":{"othermodel":"DecisionTree"}},
                "iterations":3      },

        {   "name":"ld-pca_qda-happy_dt-unhappy",
            "description":"""Apply the landmark distance feature extraction
                             and concatenate with PCA, then apply QDA to
                             find if happy then apply SVM rbf if else""",
                "preprocessing":{   "method":None,
                                    "parameters":None},

                "feature_extraction":{  "method":"LandmarkDistance+PCA",
                                        "parameters":{}},
                "classification":{   "method":"QDAplus",
                                    "parameters":{"othermodel":"SVM-RBF"}},
                "iterations":3      },


        ]

w = {   "name":"RND",
    "description":"Check baseline for random guessing of classes.",
        "preprocessing":{   "method":None,
                            "parameters":None},

        "feature_extraction":{  "method":"PCA",
                                "parameters":{"n_components":0.98}},
        "classification":{   "method":None,
                            "parameters":None},
        "iterations":1      }

CLASSES = ["happy","neutral","sad","test","angry"]
IMG_SHAPE = (250,250)
