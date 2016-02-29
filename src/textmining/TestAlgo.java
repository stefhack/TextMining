/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package textmining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.NullStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stemmers.Stemmer;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.tokenizers.Tokenizer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author Stef
 */
public class TestAlgo {

    private Classifier algo;
    private String[] options;
    private Instances data;
    private String stop_words_path_file;
    private String arff_file_path;
    private ArrayList<HashMap<String, Double>> results = new ArrayList<>();   

    public TestAlgo(String arff_file_path) {
        this.arff_file_path = arff_file_path;
    }
    
    /**
     * Cette méthode permet de filtrer les données avec la méthode StringToWordVector
     * qui décompose les textes/commentaires en mots
     * @param IDF
     * @param TF
     * @param numClass L'index de la classe à classifier
     * @param stemmer
     * @throws IOException
     * @throws Exception 
     */
    public void buildData(boolean IDF, boolean TF, int numClass,String stemmer,String tokenizer) throws IOException, Exception {

        try (BufferedReader reader = new BufferedReader(new FileReader(this.arff_file_path))) {
            this.data = new Instances(reader);
        }
        StringToWordVector wordVector = new StringToWordVector();
        wordVector.setTokenizer(getTokenizer(tokenizer));
        wordVector.setInputFormat(this.data);
        wordVector.setStopwords(new File(stop_words_path_file));
        wordVector.setTFTransform(TF);
        wordVector.setIDFTransform(IDF);
        wordVector.setStemmer(getStemmer(stemmer));
        wordVector.setWordsToKeep(10000);
        this.data = Filter.useFilter(data, wordVector);
        this.data.setClassIndex(numClass);
    }
    
    
    private Tokenizer getTokenizer(String type){
        Tokenizer tokenizer = null;
        switch(type){
            case  "Alphabetical" : tokenizer = new AlphabeticTokenizer();break;
            case "NGram" : tokenizer = new NGramTokenizer();break;
            case "Word" : tokenizer = new WordTokenizer();break;
        }
        
        return tokenizer;
    }
    
    /**
     * Permet d'avoir un Stemmer selon son nom
     * @param String type
     * @return Stemmer
     */
    public Stemmer getStemmer(String type){
        Stemmer stemmer = null ;
        switch(type){
            case "NullStemmer" : stemmer = new NullStemmer();break;
            case "LovinsStemmer" : stemmer = new LovinsStemmer();break;
            case "IteratedLovinsStemmer" : stemmer = new IteratedLovinsStemmer();break;
        }
       return stemmer;
    }
    
    public String evaluateRegression(String[] options,int nbFolds) throws Exception{
                setOptions(options);
        algo.buildClassifier(data);
        Evaluation eval = new Evaluation(data);        
        eval.crossValidateModel(algo, data, nbFolds, new Random(1));
        eval.evaluateModel(algo, data);       
        String resume = eval.toSummaryString();             
        return resume;
    }
    
    public String evaluate(String[] options,int nbFolds) throws Exception {
        String string_options = "";
        for(String option : options){
            string_options+=option;
        }
        
        setOptions(options);
        algo.buildClassifier(data);
        Evaluation eval = new Evaluation(data);        
        eval.crossValidateModel(algo, data, nbFolds, new Random(1));
        eval.evaluateModel(algo, data);       
        String resume = eval.toSummaryString();
        String matrice_confusion = "";                        
        matrice_confusion  = eval.toMatrixString();               
        HashMap<String,Double> result = new HashMap<>();
        result.put(/*"Avec les options : "+string_options !=  "" ? string_options:"aucune"+" => ALGO : "+*/String.valueOf(algo),eval.pctCorrect());
        results.add(result);
        return eval.toMatrixString(resume);
//        return resume+"\n"+matrice_confusion;
    }

    public Classifier getAlgo() {
        return algo;
    }

    public String getBestAlgo(){
        
        Double maxPercentCorrect = 0.0;
        String bestAlgo = null;
        for(HashMap<String,Double> map : results){
            for(Entry<String,Double> entry:map.entrySet()){
                String algo = entry.getKey();
                Double percentCorrect = entry.getValue();
               
                if(maxPercentCorrect < percentCorrect){
                    maxPercentCorrect = percentCorrect;
                    bestAlgo = algo;
                }
            }
        }
        
        return bestAlgo;
    }
    
    public void setAlgo(Classifier algo) {
        this.algo = algo;
    }

    public String[] getOptions() {
        return options;
    }

    public void setOptions(String[] options) {
        this.options = options;
    }

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        this.data = data;
    }

    public String getStop_words_path_file() {
        return stop_words_path_file;
    }

    public void setStop_words_path_file(String stop_words_path_file) {
        this.stop_words_path_file = stop_words_path_file;
    }

    public String getArff_file_path() {
        return arff_file_path;
    }

    public void setArff_file_path(String arff_file_path) {
        this.arff_file_path = arff_file_path;
    }

}
