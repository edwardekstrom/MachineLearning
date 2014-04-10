package decision;

import java.awt.Robot;
import java.io.PrintWriter;
import java.util.*;

import toolKit.Matrix;
import toolKit.SupervisedLearner;

public class DecisionTree extends SupervisedLearner {
	
	private static int validationPercentage = 10;
	private static int graphCount = 0;
	private static String nodeCounts = "";
	public DecisionNode _rootNode;
	private List<Double[]> _validationInstanceInputs;
	private List<Double> _validationInstanceOutputs;
	
	

	private List<DecisionNode> _leafNodes;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		_rootNode = new DecisionNode();
		_leafNodes = new ArrayList<DecisionNode>();
		_rootNode._decidionTree = this;
		_rootNode.setTitle(Double.POSITIVE_INFINITY);
		Set<Double> outputClasses = new TreeSet<Double>();
		
		for(int i = 0; i < features.cols(); i++){
			_rootNode.addAttribute(i);
		}
		
		Random rand = new Random();
		for(int i = 0; i < features.rows(); i++){
			Double[] instance = new Double[features.cols()];
			for (int j = 0; j < features.cols(); j++) {
				instance[j] = features.get(i, j);
			}
			Double output = labels.get(i, 0);
			
			if (rand.nextInt() % validationPercentage != 0) {
//			if (true) {
				_rootNode.addInstance(instance, output);
				outputClasses.add(labels.get(i, 0));
			}else{
				addValidationInstance(instance, output);
			}
			
//			System.out.println(": " + labels.get(i, 0));
		}
		
		_rootNode.setOutputClasses(outputClasses);
		
		double infoOfRoot = _rootNode.calculateInformation();
//		System.out.println(infoOfRoot);
		
		_rootNode.expandTree(_rootNode.getMajority());
		
		for (DecisionNode leaf: _leafNodes){
			leaf.pruneUp(_validationInstanceInputs, _validationInstanceOutputs);
		}
		
		_rootNode.removePrunedBranches();
		
		int nodeCount = _rootNode.countNodes();
		
		nodeCounts += nodeCount + "\n";
		
//		PrintWriter writer = new PrintWriter("votingWPruning" + ".gv");
//		writer.print("digraph voting {\n");
//		writer.print(_rootNode.makeDot());
//		writer.print("\n}");
//		writer.close();
		
//		if(graphCount == 10){
//			System.out.println(nodeCounts);
//		}
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = _rootNode.inferOutput(features);
	}
	
	public DecisionTree(){
		_rootNode = new DecisionNode();
		_validationInstanceInputs = new ArrayList<Double[]>();
		_validationInstanceOutputs = new ArrayList<Double>();

		_leafNodes = new ArrayList<DecisionNode>();
	}
	
	private void addValidationInstance(Double[] instanceInputs, Double instanceOutput){
		_validationInstanceInputs.add(instanceInputs);
		_validationInstanceOutputs.add(instanceOutput);
	}
	
	public void addLeaf(DecisionNode node){
		_leafNodes.add(node);
	}
	
	public void removeLeaf(DecisionNode node){
		_leafNodes.remove(node);
	}

}
