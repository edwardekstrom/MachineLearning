package perceptron;

import java.util.HashMap;
import java.util.Random;
import java.util.TreeSet;

import javax.swing.text.html.MinimalHTMLWriter;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class Perceptron extends SupervisedLearner{
	HashMap<Integer, Perceptron_Head> _perceptronHeads;
	Random _rand;
	public Perceptron(Random rand){
		_rand = rand;
		_perceptronHeads = new HashMap<Integer, Perceptron_Head>();
	}

	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		TreeSet<Double> classes = new TreeSet<Double>();
		for(int i = 0; i < labels.rows(); i++){
			classes.add(labels.get(i, 0));
		}
		
		for(int i = 0; i < classes.size(); i++){
			Matrix copy = new Matrix(labels, 0, 0, labels.rows(), labels.cols());
			
			for(int j = 0; j<copy.rows(); j++){
				if((int)copy.get(j, 0) == i){
					copy.set(j, 0, 1);
				}else{
					copy.set(j, 0, 0);
				}
			}
			
			Perceptron_Head ph = new Perceptron_Head(_rand);
			ph.train(features, copy);
			_perceptronHeads.put(i, ph);
		}
		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		double curHighestValue = Double.MIN_VALUE;
		double curBest = 0.0;
		for(Integer i:_perceptronHeads.keySet()){
			Perceptron_Head ph = _perceptronHeads.get(i);
			ph.predict(features, labels);
			if(labels[0] > curHighestValue) curBest = i;
		}
		labels[0] = curBest;
	}
}
