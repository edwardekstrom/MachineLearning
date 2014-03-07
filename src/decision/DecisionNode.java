package decision;

import java.text.AttributedCharacterIterator.Attribute;
import java.util.*;

public class DecisionNode {
	
	private List<Integer> _remainingAttributes;
	private List<Double[]> _instanceInputs;
	private List<Double> _instanceOutputs;
	private Set<Double> _outputClasses;
	private double _information;
	private boolean _isLeaf;
	
	private List<DecisionNode> _actualChildren;
	
	private List< List<DecisionNode> > _possibleExpansions;
	
	
	
	
	public DecisionNode() {
		_remainingAttributes = new ArrayList<Integer>();
		_instanceInputs = new ArrayList<Double[]>();
		_instanceOutputs = new ArrayList<Double>();
		_outputClasses = new TreeSet<Double>();
		_information = -1;
		_isLeaf = false;
		
		_actualChildren = new ArrayList<DecisionNode>();
		
		_possibleExpansions = new ArrayList<List<DecisionNode>>();
	}
	
	public void expandTree(){
		for(Integer attr: _remainingAttributes){
			_possibleExpansions.add(expandOn(attr));
		}
	}
	
	private List<DecisionNode> expandOn(Integer attr) {
		Map<Double, List<Double[]> > attributeToInstances = new HashMap<Double, List<Double[]> >();
		Map<Double, List<Double> > attributeToOutputs = new HashMap<Double, List<Double> >();
		
		for(int i = 0 ; i < _instanceInputs.size() ; i++){
			Double[] instanceInput = _instanceInputs.get(i);
			Double instanceOutput = _instanceOutputs.get(i);
			
			List<Double[]> instances;
			List<Double> outputs;
			
			if(attributeToInstances.containsKey(instanceInput[attr])){
				instances = attributeToInstances.get(instanceInput[attr]);
				outputs = attributeToOutputs.get(instanceInput[attr]);
			}else{
				instances = new ArrayList<Double[]>();
				outputs = new ArrayList<Double>();
			}
			
			instances.add(instanceInput);
			outputs.add(instanceOutput);
			
			attributeToInstances.put(instanceInput[attr], instances);
			attributeToOutputs.put(instanceInput[attr], outputs);
		}
		
		List<DecisionNode> possibleChildren = new ArrayList<DecisionNode>();
		
		for(Double key: attributeToInstances.keySet()){
			DecisionNode possibleChild = new DecisionNode();
			
			for(int i = 0; i < _remainingAttributes.size(); i++){
				if(_remainingAttributes.get(i) != attr){
					possibleChild.addAttribute(_remainingAttributes.get(i));
				}
			}
			
			List<Double[]> curListOfInstances = attributeToInstances.get(key);
			List<Double> curListOfOutputs = attributeToOutputs.get(key);
			
			for(int i = 0; i < curListOfInstances.size(); i++){
				Double[] instance = curListOfInstances.get(i);
				Double output = curListOfOutputs.get(i);
				
				possibleChild.addInstance(instance);
				possibleChild.addOutput(output);
//				System.out.println(": " + labels.get(i, 0));
			}
			
			possibleChild.setOutputClasses(this._outputClasses);
			
			double infoOfPossibleChild = possibleChild.calculateInformation();
			System.out.println(infoOfPossibleChild);
			
			possibleChildren.add(possibleChild);
		}
		
		return possibleChildren;
	}

	public void addAttribute(Integer attribute){
		_remainingAttributes.add(attribute);
	}
	
	public void addInstance(Double[] instance){
		_instanceInputs.add(instance);
	}
	
	public void addOutput(double output){
		_instanceOutputs.add(output);
	}
	
	public double calculateInformation(){
		Map<Double, Double> classCounts = new HashMap<Double, Double>();
		for(Double c: _outputClasses){
			classCounts.put(c, 0d);
		}
		
		for(Double curOut: _instanceOutputs){
			classCounts.put(curOut, classCounts.get(curOut) + 1);
		}
		
		double info = 0;
		
		for(Double key: classCounts.keySet()){
			if(classCounts.get(key) !=0){
				double p = classCounts.get(key) / (double)_instanceOutputs.size();
				info = info - (p * log_2(p));
			}
		}
//		if(info == 0){
//			int j = 0;
//			for(Double[] inputs: _instanceInputs){
//				for(int i = 0 ; i < inputs.length ; i++){
//					System.out.print(inputs[i] + " ");
//				}
//				System.out.println(": " + _instanceOutputs.get(j++));
//			}
//		}
		_information = info;
		return _information;
	}

	public void setOutputClasses(Set<Double> outputClasses) {
		_outputClasses = outputClasses;
	}
	
	private static double log_2(double x){
		return Math.log(x)/Math.log(2);
	}
	
	public boolean isLeaf(){
		return _isLeaf;
	}
	
	
}
