package knn;

import java.util.Comparator;

public class comp implements Comparator<double[]>{
	private double[] _instance;
	
	public comp(double[] instance){
		_instance = instance;
	}
	
	@Override
	public int compare(double[] o1, double[] o2) {
		if(InstanceBasedLearner.calculateDistance(_instance, o1) < InstanceBasedLearner.calculateDistance(_instance, o2))
			return 1;
		if(InstanceBasedLearner.calculateDistance(_instance, o1) > InstanceBasedLearner.calculateDistance(_instance, o2))
			return -1;
		return 0;
	}
	
}