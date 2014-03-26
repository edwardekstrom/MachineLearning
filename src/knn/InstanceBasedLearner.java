package knn;

import java.util.*;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class InstanceBasedLearner extends SupervisedLearner {
	ArrayList<double[]> _features;
	ArrayList<double[]> _labels;
	int _k = 13;
	static boolean REGRESSION = true;
	static boolean DISTANCE_WEIGHTED = true;
	
	public InstanceBasedLearner(){
		_features = new ArrayList<double[]>();
		_labels = new ArrayList<double[]>();
	}
	public InstanceBasedLearner(int k){
		_features = new ArrayList<double[]>();
		_labels = new ArrayList<double[]>();
		_k = k;
	}
	
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
//		_k = features.rows();
		System.out.println("original size: " + features.rows());
		for(int i = 0; i<_k; i++){
			double[] row = new double[features.cols()];
			double[] label = new double[1];
			
			for(int j = 0; j<features.cols(); j++){
				row[j] = features.get(i, j);
			}
			label[0] = labels.get(i, 0);
			_features.add(row);
			_labels.add(label);

		}
		for(int i = _k; i<features.rows(); i++){
			double[] row = new double[features.cols()];
			double[] label = new double[1];
			
			for(int j = 0; j<features.cols(); j++){
				row[j] = features.get(i, j);
			}
			label[0] = labels.get(i, 0);
			double[] prediction = new double[1];
			predict(row, prediction);
			if (Math.pow(prediction[0] - label[0],2) >.01) {
				_features.add(row);
				_labels.add(label);
			}
		}
		System.out.println("reduced size: " + _features.size());
//		System.out.println("original size: " + _features.size());
//		
//		HashSet<Integer> toRemove = new HashSet<Integer>();
//		HashSet<Integer> dontRemove = new HashSet<Integer>();
//		for (int i = 0; i < _features.size(); i++) {
//			if (!toRemove.contains(i)) {
//				for (int j = 0; j < _features.size(); j++) {
//					if (i != j && !dontRemove.contains(j) && !toRemove.contains(j)) {
//						if (calculateDistance(_features.get(i), _features.get(j)) < 1) {
//							toRemove.add(j);
//							dontRemove.add(i);
//						}
//					}
//				}
//			}
//		}
//		for(Integer i: toRemove){
//			_features.remove(i);
//			_labels.remove(i);
//		}
//		System.out.println("reduced size: " + _features.size());

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
		if (REGRESSION) {
			double sum = 0.0;
			double weights = 0.0;
			if(DISTANCE_WEIGHTED){
				for (double[] key : bestK.keySet()) {
					double curWeight = 1/Math.pow(calculateDistance(features, key), 2);
					sum += curWeight * bestK.get(key)[0];
					weights = weights + curWeight;
				}
				labels[0] = sum/weights;
			} else {
				for (double[] key : bestK.keySet()) {
					sum += bestK.get(key)[0];
				}
				labels[0] = sum/(double)bestK.size();
			}
			
		} else {
			TreeMap<Double, Double> votes = new TreeMap<Double, Double>();

			for (double[] key : bestK.keySet()) {
				double vote = 0.0;
				if (votes.containsKey(bestK.get(key)[0])) {
					vote = votes.get(bestK.get(key)[0]);
				}
				if(DISTANCE_WEIGHTED){
					vote = vote + 1/Math.pow(calculateDistance(features, key), 2);
				}else{
					vote = vote + 1;
				}
				votes.put(bestK.get(key)[0], vote);
				// System.out.println(bestK.get(key)[0] + ": " + vote);
			}

			double mostPopular = -1;
			double winner = -1.0;
			for (Double key : votes.keySet()) {
				if (votes.get(key) > mostPopular) {
					mostPopular = votes.get(key);
					winner = key;
				}
			}
			// System.out.println("winner: " + winner);
			labels[0] = winner;
		}
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


