/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package textmining;

import java.io.BufferedReader;
import weka.core.Instances;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LeastMedSq;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Stef
 */
public class TextMining {

    /**
     * Main
     *
     * @param args
     * @throws FileNotFoundException
     * @throws IOException
     * @throws Exception
     */
    public static void main(String[] args) throws IOException, Exception {

//        System.out.println("File selected : "+arff);
        /*OPTIONS*/
        String arff = "C:/wamp/www/AllocineHelper/arff/100_commentaires_1_cat.arff";
        // String arff = "/Users/Mathieu/NetBeansProjects/AllocineHelper/arff/20160104.arff";
        boolean showRegression = Boolean.valueOf("false");
        int nb_folds = 9;
        boolean setIDF = true;
        boolean setTF = true;
        String stemmer = "LovinsStemmer";
        String tokenizer = "Alphabetical";

//        String arff = args[0];
//        boolean showRegression = Boolean.valueOf(args[1]);
//        int nb_folds = Integer.valueOf(args[2]);
//        boolean setIDF = Boolean.valueOf(args[3]);
//        boolean setTF = Boolean.valueOf(args[4]);
//        String stemmer = args[5];
//        String tokenizer = args[6];
        String stopWords = "C:/wamp/www/AllocineHelper/stopwords_fr.txt";
////        String stopWords = "/Users/Mathieu/NetBeansProjects/AllocineHelper/stopwords_fr.txt";        

//        TestAlgo test1 = new TestAlgo(arff);
//        test1.setStop_words_path_file(stopWords);
//        try {
//            test1.buildData(setIDF, setTF, 1, stemmer, tokenizer);//IDF=>true/false , TF=>true/false , Classe 1 => Commentaires
//        } catch (Exception ex) {
//            System.out.println("Fichier inconnu");
//        }
//        System.out.println("*************CLASSIFICATION********************");
//        System.out.println("-------OPTIONS--------");
//        System.out.println("IDF : " + String.valueOf(setIDF));
//        System.out.println("TF : " + String.valueOf(setTF));
//        System.out.println("Nb Folds for Cross Validation : " + nb_folds);
//        System.out.println("Stemmer : " + stemmer);
//        System.out.println("Tokenizer : " + tokenizer);
//        System.out.println("-----------------------");
//        
//        System.out.println("*******************");
//        System.out.println("DECISION TABLE");
//        System.out.println("*******************");
//        Classifier decisionTable = (Classifier) new DecisionTable();
//        test1.setAlgo(decisionTable);
//        String[] options = weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 5\"");
//        System.out.println(test1.evaluate(options, nb_folds));
//
//        System.out.println("*******************");
//        System.out.println("NAIVE BAYES");
//        System.out.println("*******************");
//        Classifier naiveBayes = (Classifier) new NaiveBayes();
//        test1.setAlgo(naiveBayes);
//        System.out.println(test1.evaluate(weka.core.Utils.splitOptions(""), nb_folds));
//
//        System.out.println("*******************");
//        System.out.println("J 48");
//        System.out.println("*******************");
//        Classifier j48 = new J48();
//        test1.setAlgo(j48);
//        System.out.println(test1.evaluate(weka.core.Utils.splitOptions(""), nb_folds));
//        
//        System.out.println("*******************");
//        System.out.println("ONE R");
//        System.out.println("*******************");
//        Classifier oneR = new OneR();
//
//        test1.setAlgo(oneR);
//        System.out.println(test1.evaluate(weka.core.Utils.splitOptions(""), nb_folds));
//
////        System.out.println("And the winner is : " + test1.getBestAlgo());
//        if (showRegression) {
//
//            HashMap<String, Classifier> regressionClassifiers = new HashMap<String, Classifier>();
//
//            regressionClassifiers.put("LinearRegression", (Classifier) new LinearRegression());
//            regressionClassifiers.put("SMO Reg", (Classifier) new SMOreg());
//            regressionClassifiers.put("LeastMedSq", new LeastMedSq());
//            System.out.println("***********************REGRESSION****************************");
//
//            for (Map.Entry<String, Classifier> entry : regressionClassifiers.entrySet()) {
//                System.out.println("Algo : " + entry.getKey());
//                Classifier algo = entry.getValue();
//                test1.setAlgo(algo);
//                test1.buildData(setIDF, setTF, 0, stemmer, tokenizer);
//                options = weka.core.Utils.splitOptions("");
//                System.out.println(test1.evaluateRegression(options, nb_folds));
//                System.out.println(algo);
//            }
//
//        }
        //Tests predictions
//        TestAlgo prediction = new TestAlgo(arff);
//        prediction.setStop_words_path_file(stopWords);
//        Classifier algo =   (Classifier)new NaiveBayes();
//        prediction.setAlgo(algo);
//         prediction.buildData(setIDF, setTF, 1, stemmer,tokenizer);
//        
//          String[] options = weka.core.Utils.splitOptions("");
//          prediction.evaluateRegression(options, nb_folds);
//         weka.core.SerializationHelper.write("naive_bayes.model", algo);
//System.exit(-1);
        Instances data;
//            LinearRegression LR = (LinearRegression)weka.core.SerializationHelper.read("linear_reg.model");
        NaiveBayes NB = (NaiveBayes) weka.core.SerializationHelper.read("naive_bayes.model");
        try (BufferedReader reader = new BufferedReader(new FileReader("C:/wamp/www/AllocineHelper/arff/supplied_test.arff"))) {
            data = new Instances(reader);
        }
        data.setClassIndex(data.numAttributes()-1);
        int nb_good =0;
        for (int i = 0; i < data.numInstances(); i++) {
            double actualValue = data.instance(i).classValue();
          
            Instance newInst = data.instance(i);
            System.out.println(newInst);
            double predAlgo = NB.classifyInstance(newInst);
            if(actualValue == predAlgo) nb_good++;
            System.out.println(data.classAttribute().value((int) actualValue) + "  => " + data.classAttribute().value((int) predAlgo));
            System.out.println("***********************");
        }
       
        System.out.println("Pourcentage réussite prédictions : "+((double)nb_good/(double)data.numInstances()*100)+ " %");
//         for (int i = 0; i < data.numInstances(); i++) {
//            double actualValue = data.instance(i).classValue();
//             System.out.println(actualValue);
//            Instance newInst = data.instance(i);
//            
//            double predAlgo = LR.classifyInstance(newInst);
//             System.out.println(actualValue + "  => "+predAlgo);
//        }
//        
//        System.out.println("OPTIONS FILTER : STOPWORDS ONLY");
//        // C_DecisionTable
//        String rep_DecisionTable = C_DecisionTable(dataFiltered);
//        System.out.println("DECISION TABLE");
//        System.out.println(rep_DecisionTable);
//    
//        // ---------------
//        System.out.println("\n--------------------\n");
//        
//        // C_NaiveBayes
//        String rep_NaiveBayes = C_NaiveBayes(dataFiltered);
//        System.out.println("NAIVE BAYES");
//        System.out.println(rep_NaiveBayes);
//        //                       
//        
//        System.out.println("OPTIONS FILTER : STOPWORDS + IDF/TF TRANSFORM");
//        
//        StringToWordVector wordVector2 = new StringToWordVector();
//        wordVector2.setInputFormat(data);        
//        wordVector2.setStopwords(new File(stopWords));
//        wordVector2.setIDFTransform(true);
//        wordVector2.setTFTransform(true);
////        wordVector2.setStemmer(new SnowballStemmer());
//        Instances dataFilteredWithOptions = Filter.useFilter(data, wordVector2);
//        dataFilteredWithOptions.setClassIndex(1);
//        
//         System.out.println("LINEAR REGRESSION ON POLARITY");
//         dataFiltered.setClassIndex(0);
//        String result_Regression = Regression_on_Polarity(dataFiltered);
//        System.out.println(result_Regression);
//        
//        String rep_DecisionTable_with_Options = C_DecisionTable(dataFilteredWithOptions);        
//        System.out.println("DECISION TABLE");
//        System.out.println(rep_DecisionTable_with_Options);
//        
//        // C_NaiveBayes
//        String rep_NaiveBayes_with_Options = C_NaiveBayes(dataFilteredWithOptions);
//        System.out.println("NAIVE BAYES");
//        System.out.println(rep_NaiveBayes_with_Options);
//        
    }

    /**
     * Decision Table
     *
     * @param instances
     * @return string
     * @throws Exception
     */
    private static String C_DecisionTable(Instances instances) throws Exception {
        Classifier decisionTable = (Classifier) new DecisionTable();
        String[] options = weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 5\"");
        decisionTable.setOptions(options);
        decisionTable.buildClassifier(instances);
        Evaluation eval = new Evaluation(instances);
//        eval.evaluateModel(decisionTable, instances);
        eval.crossValidateModel(decisionTable, instances, 5, new Random(1));
        String resume = eval.toSummaryString();

        return eval.toMatrixString(resume);
    }

    /**
     * Naive Bayes
     *
     * @param instances
     * @return string
     * @throws Exception
     */
    private static String C_NaiveBayes(Instances instances) throws Exception {
        Classifier naiveBayes = (Classifier) new NaiveBayes();
        String[] options = weka.core.Utils.splitOptions("");
        return setOptions(naiveBayes, instances, options);
    }

    private static String Regression_on_Polarity(Instances instances) throws Exception {
        Classifier regression = (Classifier) new LinearRegression();
        String[] options = weka.core.Utils.splitOptions("-S 2 -R 1.0E-8");
        return setOptions(regression, instances, options);
    }

    private static String setOptions(Classifier classifier, Instances instances, String[] options) throws Exception {
        classifier.setOptions(options);
        classifier.buildClassifier(instances);
        Evaluation eval = new Evaluation(instances);
        eval.crossValidateModel(classifier, instances, 5, new Random(1));
        eval.evaluateModel(classifier, instances);
        String resume = eval.toSummaryString();
        return eval.toMatrixString(resume);
    }

}
