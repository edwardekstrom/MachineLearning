package knn;

import java.util.*;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class InstanceBasedLearner extends SupervisedLearner {
	ArrayList<double[]> _features;
	ArrayList<double[]> _labels;
	int _k;
	
	public InstanceBasedLearner(){
		_features = new ArrayList<double[]>();
		_labels = new ArrayList<double[]>();
		_k = 3;
	}
	
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		for(int i = 0; i<features.rows(); i++){
			double[] row = new double[features.cols()];
			double[] label = new double[1];
			
			for(int j = 0; j<features.cols(); j++){
				row[j] = features.get(i, j);
			}
			label[0] = labels.get(i, 0);
			_features.add(row);
			_labels.add(label);
		}
	}
	
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		Comparator<double[]> comp = new comp(features);
		TreeMap<double[],double[]> bestK = new TreeMap<double[], double[]>(comp);
		double furthest;
		for(int i = 0; i<_k; i++){
			bestK.put(_features.get(i), _labels.get(i));
		}
		furthest = calculateDistance(features, bestK.firstKey());
		
		for(int i = _k; i<_features.size(); i++){
			double curDist = calculateDistance(features, _features.get(i));
			if(curDist < furthest){
				bestK.remove(bestK.firstKey());
				bestK.put(_features.get(i), _labels.get(i));
				furthest = calculateDistance(features, bestK.firstKey());
				
//				System.out.println("update:\n\n");
//				for(double[] key: bestK.keySet()){
//					System.out.println(calculateDistance(features, key));
//					
//				}
			}
//			System.out.println(furthest);
		}
		
		TreeMap<Double, Double> votes = new TreeMap<Double, Double>();
		
		for(double[] key: bestK.keySet()){
			double vote = 0.0;
			if(votes.containsKey(bestK.get(key)[0])){
				vote = votes.get(bestK.get(key)[0]);
			}
			vote = vote+1;
			votes.put(bestK.get(key)[0], vote);
//			System.out.println(bestK.get(key)[0] + ": " + vote);
		}
		
		double mostPopular = -1;
		double winner = -1.0;
		for(Double key: votes.keySet()){
			if(votes.get(key) > mostPopular){
				mostPopular = votes.get(key);
				winner = key;
			}
		}
//		System.out.println("winner: " + winner);
		labels[0] = winner;
	}
	
	public static double calculateDistance(double[] array1, double[] array2)
    {
        double Sum = 0.0;
        for(int i=0;i<array1.length;i++) {
           Sum = Sum + Math.pow((array1[i]-array2[i]),2.0);
        }
        return Math.sqrt(Sum);
    }
	
}


