package decision;

import java.util.*;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class DecisionTree extends SupervisedLearner {
	
	
	private DecisionNode _rootNode;
	
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		Set<Double> outputClasses = new TreeSet<Double>();
		
		for(int i = 0; i < features.cols(); i++){
			_rootNode.addAttribute(i);
		}
		
		for(int i = 0; i < features.rows(); i++){
			Double[] instance = new Double[features.cols()];
			for(int j = 0; j < features.cols(); j++){
				instance[j] = features.get(i, j);
//				System.out.print(features.get(i, j) + " ");
			}
			_rootNode.addInstance(instance);
			_rootNode.addOutput(labels.get(i, 0));
			outputClasses.add(labels.get(i, 0));
			
//			System.out.println(": " + labels.get(i, 0));
		}
		
		_rootNode.setOutputClasses(outputClasses);
		
		double infoOfRoot = _rootNode.calculateInformation();
		System.out.println(infoOfRoot);
		
		_rootNode.expandTree();
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = 0d;
	}
	
	public DecisionTree(){
		_rootNode = new DecisionNode();
	}

}
