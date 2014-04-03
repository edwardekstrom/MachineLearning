package clustering;


import java.util.*;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class Kmeans extends SupervisedLearner {
	private Map<Centroid, List<double[]>> _centroidToCluster;
	private List<double[]> _instances;
	private Set<Integer> _used;
	private static int K = 7;
	private Matrix _features;
	
	
	
	public Kmeans(){
//		System.out.println(Double.MAX_VALUE);
		_centroidToCluster = new TreeMap<Centroid, List<double[]>>();
		_instances = new ArrayList<double[]>();
		_used = new HashSet<Integer>();
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		_features = features;
		for(int i = 0; i<features.rows(); i++){
			double[] row = new double[features.cols()];
			
			for(int j = 0; j<features.cols(); j++){
				row[j] = features.get(i, j);
			}
			_instances.add(row);
		}
		
		for(int i = 0; i < K; i++){
			Random rand = new Random();
			int r = rand.nextInt();
			r = r % (_instances.size() - 1);
			while(_used.contains(r)){
				r = rand.nextInt();
				r = r % (_instances.size() - 1);
			}
			if(r<0) r = r * -1;
			double[] row = new double[features.cols()];
			
			for(int j = 0; j<features.cols(); j++){
				row[j] = features.get(r, j);
//				row[j] = features.get(i, j);
			}
			Centroid c = new Centroid();
			c.array = row;
			c.name = new Integer(i).toString();
			_centroidToCluster.put(c, new ArrayList<double[]>());
		}
		
		makeClusters();
	}

	private void makeClusters() {
		boolean hasChanged = true;
		
		while(hasChanged){
//			double ssd = 0;
			hasChanged = false;
			String sizesString = "";
			String SSEs = "";
			for(Centroid key: _centroidToCluster.keySet()){
				System.out.println("Centroid " + key.name + " : " + printArray(key.array));
			}
			
			for(int i = 0; i < _instances.size(); i++){
				double[] curInstance = _instances.get(i);
				Centroid closestCentroid = _centroidToCluster.keySet().iterator().next();
				double closestDistance = calculateDistance(curInstance, closestCentroid.array);
				
				for(Centroid key: _centroidToCluster.keySet()){
					double curDistance = calculateDistance(curInstance, key.array);
					if(curDistance < closestDistance){
						closestCentroid = key;
						closestDistance = curDistance;
					}
				}
				if( i % 10 != 0) System.out.print(i + "=" + closestCentroid.name + ", ");
				else System.out.println(i + "=" + closestCentroid.name + ",");
				_centroidToCluster.get(closestCentroid).add(curInstance);
			}
			System.out.println();
			
			for(Centroid key: _centroidToCluster.keySet()){
				sizesString += "Centroid " + key.name + ": " + _centroidToCluster.get(key).size() +"\n";
				List<double[]> curInstances = _centroidToCluster.get(key);
				for(int j = 0; j<key.array.length; j++){
					if(_features._columsAreNumbers.get(j)){
						double total = 0;
						int questionMarks = 0;
						for(int i = 0 ; i < curInstances.size(); i++){
							if(curInstances.get(i)[j] == Double.MAX_VALUE){
								questionMarks++;
							}else{
								total += curInstances.get(i)[j];
							}
						}
						int notQuestionMarks = (curInstances.size() - questionMarks);
						double keyNewJValue = total/(double)notQuestionMarks;
						if(notQuestionMarks == 0 && key.array[j] != Double.MAX_VALUE){
							hasChanged = true;
							key.array[j] = Double.MAX_VALUE;
						} else {
							if (Math.pow(keyNewJValue - key.array[j], 2) > 0) {
								hasChanged = true;
								key.array[j] = keyNewJValue;
							}
						}
					}else{
						Map<Double, Integer> valueToCount = new TreeMap<Double, Integer>();
						for(int i = 0; i < curInstances.size(); i++){
							double iJthIndexValue = curInstances.get(i)[j];
							if(valueToCount.containsKey(iJthIndexValue)){
								int newCount = valueToCount.get(iJthIndexValue);
								newCount++;
								valueToCount.put(iJthIndexValue, newCount);
							}else{
								valueToCount.put(iJthIndexValue, 1);
							}
						}
						double mostCommonValue = -1;
						int commonValueCount = -1;
						for(double value: valueToCount.keySet()){
							if(valueToCount.get(value) > commonValueCount){
								mostCommonValue = value;
								commonValueCount = valueToCount.get(value);
							}
						}
						if(key.array[j] != mostCommonValue){
							hasChanged = true;
							key.array[j] = mostCommonValue;
						}
					}
					
				}
				SSEs += "cluster " + key.name + " SSE : " + measureCluster(_centroidToCluster.get(key)) + "\n";
			}
			int i = 0;

			System.out.print(sizesString);
			System.out.print(SSEs);
			try{
				System.out.println(measureAccuracy(_features, _features, null));
			}catch (Exception e){}
			System.out.println( "----------------------------end----------------------------");
			for(Centroid key: _centroidToCluster.keySet()){
//				System.out.print(i + "=" + _centroidToCluster.get(key).size() + " ");
				_centroidToCluster.put(key, new ArrayList<double[]>());
				i++;
			}
		}
	}
	
	double measureCluster(List<double[]> list){
		double sse = 0.0;
		for (int i = 0; i < list.size(); i++) {
			double[] feat = list.get(i);
			double[] targ = list.get(i).clone();
			
			
			try {
				predict(feat, targ);
			} catch (Exception e) {
				e.printStackTrace();
			}
			sse += Math.pow(this.calculateDistance(feat, targ), 2);
		}
		return sse;
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

	@Override
	public void predict(double[] features, double[] labels) throws Exception {

		double[] curInstance = features;
		Centroid closestCentroid = _centroidToCluster.keySet().iterator().next();
		double closestDistance = calculateDistance(curInstance, closestCentroid.array);

		for (Centroid key : _centroidToCluster.keySet()) {
			double curDistance = calculateDistance(curInstance, key.array);
			if (curDistance < closestDistance) {
				closestCentroid = key;
				closestDistance = curDistance;
			}
		}
		
		for(int j = 0; j < closestCentroid.array.length ; j++){
			labels[j] = closestCentroid.array[j];
		}

	}

	@Override
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
//@list_memoized
//def euclidean_distance(one,two):
//total = 0
//for x,y in zip(one,two):
//if is_number(x) and is_number(y):
//total += real_subdistance(x,y)
//else:
//total += nominal_subdistance(x,y)
//return total ** .5
//
//
//def real_subdistance(one,two):
//return (float(one) - float(two))**2
//
//
//def nominal_subdistance(one,two):
//if one == '?' and two == '?':
//return 1
//elif one == two:
//return 0
//else:
//return 1






















