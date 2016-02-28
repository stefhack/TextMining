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
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.rules.DecisionTable;
import weka.core.Debug.Random;
import weka.core.Instance;

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
        String arff = "C:/wamp/www/AllocineHelper/arff/50_com_plus_cat.json17_09_34.arff";
        // String arff = "/Users/Mathieu/NetBeansProjects/AllocineHelper/arff/20160104.arff";
        boolean showRegression = Boolean.valueOf("true");
        int nb_folds = 5;
        boolean setIDF = true;
        boolean setTF = true;
        String stemmer = "NullStemmer";

//        String arff = args[0];
//        boolean showRegression = Boolean.valueOf(args[1]);
//        int nb_folds = Integer.valueOf(args[2]);
//        boolean setIDF = Boolean.valueOf(args[3]);
//        boolean setTF = Boolean.valueOf(args[4]);
//         String stemmer = args[5];
        String stopWords = "C:/wamp/www/AllocineHelper/stopwords_fr.txt";
////        String stopWords = "/Users/Mathieu/NetBeansProjects/AllocineHelper/stopwords_fr.txt";        

//        TestAlgo test1 = new TestAlgo(arff);
//        test1.setStop_words_path_file(stopWords);
//        try {
//            test1.buildData(setIDF, setTF, 1, stemmer);//IDF=>true/false , TF=>true/false , Classe 1 => Commentaires
//        } catch (Exception ex) {
//            System.out.println("Fichier inconnu");
//        }
//        System.out.println("*************CLASSIFICATION********************");
//        System.out.println("DECISION TABLE");
//        Classifier decisionTable = (Classifier) new DecisionTable();
//        test1.setAlgo(decisionTable);
//        String[] options = weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 5\"");
//        System.out.println(test1.evaluate(options, nb_folds));
//
//        System.out.println("NAIVE BAYES");
//        Classifier naiveBayes = (Classifier) new NaiveBayes();
//        test1.setAlgo(naiveBayes);
//        System.out.println(test1.evaluate(weka.core.Utils.splitOptions(""), nb_folds));
//
//        System.out.println("J 48");
//        Classifier j48 = new J48();
//        test1.setAlgo(j48);
//        System.out.println(test1.evaluate(weka.core.Utils.splitOptions(""), nb_folds));
//
//        System.out.println("ONE R");
//        Classifier oneR = new OneR();
//
//        test1.setAlgo(oneR);
//        System.out.println(test1.evaluate(weka.core.Utils.splitOptions(""), nb_folds));
//
////        System.out.println("And the winner is : " + test1.getBestAlgo());
//        if (showRegression) {
//            
//            HashMap<String,Classifier> regressionClassifiers = new HashMap<String, Classifier>();
//            
//            regressionClassifiers.put("LinearRegression",(Classifier)new LinearRegression());
//            regressionClassifiers.put("SMO Reg",(Classifier)new SMOreg());
//            regressionClassifiers.put("LeastMedSq",new LeastMedSq());
//            System.out.println("***********************REGRESSION****************************");
//            
//            for (Map.Entry<String, Classifier> entry : regressionClassifiers.entrySet()) {
//                System.out.println("Algo : "+ entry.getKey());
//                Classifier algo = entry.getValue();
//                 test1.setAlgo(algo);
//                test1.buildData(setIDF, setTF, 0, stemmer);
//                options = weka.core.Utils.splitOptions("");
//                System.out.println(test1.evaluateRegression(options, nb_folds));
//System.out.println(algo);
//            }
//          
//
//        }
        
        //Tests predictions
//        TestAlgo prediction = new TestAlgo(arff);
//        prediction.setStop_words_path_file(stopWords);
//        Classifier algo =   (Classifier)new LinearRegression();
//        prediction.setAlgo(algo);
//         prediction.buildData(setIDF, setTF, 0, stemmer);
//         Instances data = prediction.getData();
//          String[] options = weka.core.Utils.splitOptions("");
//          prediction.evaluateRegression(options, nb_folds);
//         weka.core.SerializationHelper.write("linear_reg.model", algo);

            Instances data;
            LinearRegression LR = (LinearRegression)weka.core.SerializationHelper.read("linear_reg.model");
            
            try (BufferedReader reader = new BufferedReader(new FileReader("C:/wamp/www/AllocineHelper/arff/supplied_test.arff"))) {
            data = new Instances(reader);
        }
            data.setClassIndex(0);
         double actualValue = data.instance(0).classValue();
             System.out.println(actualValue);
            Instance newInst = data.instance(0);
            
            double predAlgo = LR.classifyInstance(newInst);
             System.out.println(actualValue + "  => "+predAlgo);
          

//         for (int i = 0; i < data.numInstances(); i++) {
//            double actualValue = data.instance(i).classValue();
//             System.out.println(actualValue);
//            Instance newInst = data.instance(i);
//            
//            double predAlgo = LR.classifyInstance(newInst);
//             System.out.println(actualValue + "  => "+predAlgo);
//        }
//        Instances data;
//        try (BufferedReader reader = new BufferedReader(new FileReader(arff))) {
//            data = new Instances(reader);
//        }
//
//        StringToWordVector wordVector = new StringToWordVector();
//        wordVector.setInputFormat(data);        
//        wordVector.setStopwords(new File(stopWords));
//        Instances dataFiltered = Filter.useFilter(data, wordVector);
//        dataFiltered.setClassIndex(1);
//
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
