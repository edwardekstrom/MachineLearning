package clustering;

import java.util.*;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class HAC extends SupervisedLearner {
	public double[][] _distanceMatrix;
	public Matrix _features;
	private double[][] _featuresArray;
	private String[] _names;
	private LinkInterface _link = new SingleLink();
	
	public HAC() {
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		_names = new String[features.rows()];
		for(int i = 0; i < _names.length; i++){
			_names[i] = "cluster : " + i; 
		}
		_features = features;
		_featuresArray = features.getTwoDemensionalArray();
		_distanceMatrix = new double[features.rows()][features.rows()];
		populateDistanceMatrix();
		cluster();
		

	}
	
	private void populateDistanceMatrix(){
		for(int i = 0 ; i < _distanceMatrix.length ; i++){
			for(int j = 0 ; j < _distanceMatrix.length ; j++){
				_distanceMatrix[i][j] = calculateDistance(_featuresArray[i], _featuresArray[j]);
			}
		}
	}
	
	private Cluster cluster(){
		List<Cluster> clusters = makeClusters();
		List<Pair> pairs = makePairs(clusters);
		String sses = "";
		while(clusters.size() > 1){
			joinClusters(clusters, pairs);
			double dist = -1;
			for(Cluster c: clusters){
//				System.out.println(c._name);
				if(c._distance > dist) dist = c._distance;
			}
//			System.out.println("-------------------------------" + "info" + "-------------------------------");
			if(clusters.size() > 1 && clusters.size() < 8){
				String cents = "";
				String SSEs = "";
				String counts = "";
				int clusCount = 0;
				double totalSSE = 0.0;
				for(Cluster c: clusters){
					String[] indeces = c._name.split("\\s*(=>|,|\\s)\\s*");
					ArrayList<double[]> instancesInCluster = new ArrayList<double[]>();
					for(int i = 2; i < indeces.length; i++){
						instancesInCluster.add(_featuresArray[Integer.parseInt(indeces[i])]);
					}
					double[] centroid = getCentroid(instancesInCluster);
					cents+="Cluster " + clusCount + " Centroid : " + printArray(centroid) + "\n";
					double sse = 0.0;
					for (int i = 0; i < instancesInCluster.size(); i++) {
						double[] feat = instancesInCluster.get(i);
						double[] targ = centroid.clone();

						try {
							predict(feat, targ);
						} catch (Exception e) {
							e.printStackTrace();
						}
						sse += Math.pow(this.calculateDistance(feat, targ), 2);
					}
					totalSSE += sse;
					SSEs +="Cluster " + clusCount + " SSE : " + sse + "\n";
					counts +="Cluster " + clusCount + " Count : " + instancesInCluster.size() + "\n";
//					System.out.println(Arrays.toString(indeces));
//					System.out.println(instancesInCluster.size());
					clusCount++;
				}
				System.out.print(cents);
				System.out.print(SSEs);
				System.out.print(counts);
				System.out.println("Total SSE : " + totalSSE);
				sses = "\n" + totalSSE + sses;
			}
//			System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + dist + "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^");
		}
		if(clusters.size()>1) System.out.println("!!!!!!!!!!!!!!!!!!ERRORRRRRRRRRRR!!!!!!!!!!!!!!!!!!!!!!!!clusters size greater than 1");
		System.out.println(sses);
		return clusters.get(0);
	}
	
	private String printArray(double[] array){
		String s = "";
		for(int j = 0; j < array.length; j++){
			if (array[j] == Double.MAX_VALUE)
				s += "?" + ", ";
			else if (_features._columsAreNumbers.get(j))
				s += array[j] + ", ";
			else
				s += _features.attrValue(j, (int)array[j])  + ", ";
		}
		return s;
	}
	
	private double[] getCentroid(List<double[]> curInstances) {
		double[] key = new double[curInstances.get(0).length];
		for (int j = 0; j < key.length; j++) {
			if (_features._columsAreNumbers.get(j)) {
				double total = 0;
				int questionMarks = 0;
				for (int i = 0; i < curInstances.size(); i++) {
					if (curInstances.get(i)[j] == Double.MAX_VALUE) {
						questionMarks++;
					} else {
						total += curInstances.get(i)[j];
					}
				}
				int notQuestionMarks = (curInstances.size() - questionMarks);
				double keyNewJValue = total / (double) notQuestionMarks;
				if (notQuestionMarks == 0 && key[j] != Double.MAX_VALUE) {
					key[j] = Double.MAX_VALUE;
				} else {
					if (Math.pow(keyNewJValue - key[j], 2) > 0) {
						key[j] = keyNewJValue;
					}
				}
			} else {
				Map<Double, Integer> valueToCount = new TreeMap<Double, Integer>();
				for (int i = 0; i < curInstances.size(); i++) {
					double iJthIndexValue = curInstances.get(i)[j];
					if (valueToCount.containsKey(iJthIndexValue)) {
						int newCount = valueToCount.get(iJthIndexValue);
						newCount++;
						valueToCount.put(iJthIndexValue, newCount);
					} else {
						valueToCount.put(iJthIndexValue, 1);
					}
				}
				double mostCommonValue = -1;
				int commonValueCount = -1;
				for (double value : valueToCount.keySet()) {
					if (valueToCount.get(value) > commonValueCount) {
						mostCommonValue = value;
						commonValueCount = valueToCount.get(value);
					}
				}
				if (key[j] != mostCommonValue) {
					key[j] = mostCommonValue;
				}
			}
		}
		return key;
	}

	private void joinClusters(List<Cluster> clusters, List<Pair> pairs) {
		Collections.sort(pairs);
		
		if(pairs.size() > 0){
			Pair minPair = pairs.remove(0);
			Cluster leftMerged = minPair._left;
			clusters.remove(leftMerged);
			Cluster rightMerged = minPair._right;
			clusters.remove(rightMerged);
			Cluster merged = minPair.joinClusters();
			
			for(Cluster c : clusters){
				Pair leftPair = find(c,leftMerged, pairs);
				Pair rightPair = find(c, rightMerged, pairs);
				Pair newPair = new Pair();
				newPair._left = c;
				newPair._right = merged;
				Collection<Double> distVals = new ArrayList<Double>();
				if(leftPair != null){
					distVals.add(leftPair._linkDist);
					pairs.remove(leftPair);
				}
				if(rightPair != null){
					distVals.add(rightPair._linkDist);
					pairs.remove(rightPair);
				}
				Double newDist = _link.calculateDistance(distVals);
				newPair._linkDist = newDist;
				pairs.add(newPair);
			}
			clusters.add(merged);
		}
		
	}

	private Pair find(Cluster cluster, Cluster other, List<Pair> pairs) {
		for(Pair p : pairs){
			if(p._left.equals(cluster) && p._right.equals(other) || p._left.equals(other) && p._right.equals(cluster)){
				return p;
			}
		}
		return null;
	}

	private List<Pair> makePairs(List<Cluster> clusters) {
		List<Pair> pairs = new ArrayList<Pair>();
		for(int j = 0; j < clusters.size(); j ++){
			for (int i = j+1; i < clusters.size(); i++){
				Pair pair = new Pair();
				pair._linkDist = _distanceMatrix[j][i];
				pair._left = clusters.get(j);
				pair._right = clusters.get(i);
				pairs.add(pair);
			}
		}
		return pairs;
	}

	private List<Cluster> makeClusters() {
		List<Cluster> clusters = new ArrayList<Cluster>();
		for(String name: _names){
			Cluster clust = new Cluster();
			clust._name = name;
			clusters.add(clust);
		}
		return clusters;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub

	}
	
	
	public double calculateDistance(double[] array1, double[] array2){
        double Sum = 0.0;
		for (int i = 0; i < array1.length; i++) {
			if (array1[i] == Double.MAX_VALUE || array2[i] == Double.MAX_VALUE) {
				Sum++;
			}else if(_features._columsAreNumbers.get(i)){
				Sum = Sum + Math.pow((array1[i] - array2[i]), 2.0);
			}else{
				if(array1[i] != array2[i]) Sum++;
			}
			
		}
        return Math.sqrt(Sum);
    }
}
