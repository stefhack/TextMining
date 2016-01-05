/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package textmining;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.List;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;


/**
 *
 * @author Stef
 */
public class TextMining {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */

    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
        BufferedReader reader = new BufferedReader(new FileReader("C:/wamp/www/AllocineHelper/arff/20160104.arff"));
        Instances data = new Instances(reader);
        reader.close();

        StringToWordVector wordVector = new StringToWordVector();
        wordVector.setInputFormat(data);
        Instances dataFiltered = Filter.useFilter(data, wordVector);
        dataFiltered.setClassIndex(1);

        Classifier decisionTable = (Classifier) new DecisionTable();
        String[] options = weka.core.Utils.splitOptions("-X 1 -S \"weka.attributeSelection.BestFirst -D 1 -N 5\"");
        decisionTable.setOptions(options);
        decisionTable.buildClassifier(dataFiltered);

        Evaluation eval = new Evaluation(dataFiltered);
        eval.evaluateModel(decisionTable, dataFiltered);
        String resume = eval.toSummaryString();
        String matrique = eval.toMatrixString(resume);
        System.out.println(resume);
        System.out.println(matrique);

    }
}
