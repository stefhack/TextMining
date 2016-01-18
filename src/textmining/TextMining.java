/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package textmining;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.DecisionTable;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Stef
 */
public class TextMining {

    
    /**
     * Main
     * @param args
     * @throws FileNotFoundException
     * @throws IOException
     * @throws Exception 
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        //String arff = "C:/wamp/www/AllocineHelper/arff/20160104.arff";
        String arff = "/Users/Mathieu/NetBeansProjects/AllocineHelper/arff/20160104.arff";
        // String stopWords = "C:/wamp/www/AllocineHelper/stopwords_fr.txt";
        String stopWords = "/Users/Mathieu/NetBeansProjects/AllocineHelper/stopwords_fr.txt";

        Instances data;
        try (BufferedReader reader = new BufferedReader(new FileReader(arff))) {
            data = new Instances(reader);
        }

        StringToWordVector wordVector = new StringToWordVector();
        wordVector.setInputFormat(data);
        wordVector.setStopwords(new File(stopWords));
        Instances dataFiltered = Filter.useFilter(data, wordVector);
        dataFiltered.setClassIndex(1);

        // C_DecisionTable
        String rep_DecisionTable = C_DecisionTable(dataFiltered);
        System.out.println("DECISION TABLE");
        System.out.println(rep_DecisionTable);
    
        // ---------------
        System.out.println("\n--------------------\n");
        
        // C_NaiveBayes
        String rep_NaiveBayes = C_NaiveBayes(dataFiltered);
        System.out.println("NAIVE BAYES");
        System.out.println(rep_NaiveBayes);
    }

    /**
     * Decision Table
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
        eval.evaluateModel(decisionTable, instances);
        String resume = eval.toSummaryString();
        return eval.toMatrixString(resume);
    }

    /**
     * Naive Bayes
     * @param instances
     * @return string
     * @throws Exception 
     */
    private static String C_NaiveBayes(Instances instances) throws Exception {
        Classifier naiveBayes = (Classifier) new NaiveBayes();
        String[] options = weka.core.Utils.splitOptions("");
        naiveBayes.setOptions(options);
        naiveBayes.buildClassifier(instances);
        Evaluation eval = new Evaluation(instances);
        eval.evaluateModel(naiveBayes, instances);
        String resume = eval.toSummaryString();
        return eval.toMatrixString(resume);
    }
}
