package decision;

import java.util.*;

public class DecisionNode {
	
	private List<Integer> _remainingAttributes;
	private List<Double[]> _instanceInputs;
	private List<Double> _instanceOutputs;
	private int _countInstances;
	private Set<Double> _outputClasses;
	private Map<Double, Integer> _majorityTraker;
	private double _information;
	private boolean _isLeaf;
	private boolean _isPure;
	
	private Integer _expandedOnAttribute;
	
	private Map<Double, DecisionNode> _actualChildren;
	
	private Map<Integer, Map<Double, DecisionNode> > _possibleExpansions;
	
	private Double _nodeOutput;
	
	private Double _title;
	
	
	
	public DecisionNode() {
		_remainingAttributes = new ArrayList<Integer>();
		_instanceInputs = new ArrayList<Double[]>();
		_instanceOutputs = new ArrayList<Double>();
		_countInstances = 0;
		_outputClasses = new TreeSet<Double>();
		_majorityTraker = new HashMap<Double, Integer>();
		_information = -1;
		_isLeaf = false;
		
		_actualChildren = new HashMap<Double, DecisionNode>();
		
		_possibleExpansions = new HashMap<Integer, Map<Double, DecisionNode>>();
		
		_isPure = true;
		_expandedOnAttribute = null;
		_nodeOutput = null;
	}
	
	public void expandTree(Double parentOutput){
		if(_countInstances == 0){
			for(int i = 0 ; i < _countInstances ; i++){
				for(int j = 0 ; j < _instanceInputs.get(i).length ; j++){
					System.out.print(_instanceInputs.get(i)[j] + " ");
				}
				System.out.println(": " + _instanceOutputs.get(i));
			}
			System.out.print("\n\n\n");
			
			_isLeaf = true;
			_nodeOutput = parentOutput;
			return;
		}else if(_isPure){
			for(int i = 0 ; i < _countInstances ; i++){
				for(int j = 0 ; j < _instanceInputs.get(i).length ; j++){
					System.out.print(_instanceInputs.get(i)[j] + " ");
				}
				System.out.println(": " + _instanceOutputs.get(i));
			}
			System.out.print("\n\n\n");
			
			_isLeaf = true;
			_nodeOutput = getMajority();
			return;
		}else if(_remainingAttributes.size() == 0){
			for(int i = 0 ; i < _countInstances ; i++){
				for(int j = 0 ; j < _instanceInputs.get(i).length ; j++){
					System.out.print(_instanceInputs.get(i)[j] + " ");
				}
				System.out.println(": " + _instanceOutputs.get(i));
			}
			System.out.print("\n\n\n");
			
			_isLeaf = true;
			_nodeOutput = getMajority();
			return;
		}else{
			for(Integer attr: _remainingAttributes){
				_possibleExpansions.put(attr, expandOn(attr));
			}
			
			_actualChildren = getBestGain();
			
			_nodeOutput = getMajority();
			for(Double key: _actualChildren.keySet()){
				DecisionNode actualChild = _actualChildren.get(key);
				actualChild.expandTree(_nodeOutput);
			}
		}
		
		
	}
	
	public Double getMajority() {
		Double majority = -1d;
		int highest = -1;
		for(Double key: _majorityTraker.keySet()){
			if(_majorityTraker.get(key)>highest){
				highest = _majorityTraker.get(key);
				majority = key;
			}
		}
//		System.out.println(majority + "\n\n");
		return majority;
	}

	private Map<Double, DecisionNode> getBestGain() {
		double lowest = Double.POSITIVE_INFINITY;
		Map<Double, DecisionNode> bestA = null;
		for(Integer key: _possibleExpansions.keySet()){
			Map<Double, DecisionNode> possible = _possibleExpansions.get(key);
			double curInfo_A = 0;
			
			for(Double keyPos: possible.keySet()){
				DecisionNode child = possible.get(keyPos);
				curInfo_A += (child.getCountInstances() / _countInstances) * child.getInfo();
			}
			
			if(curInfo_A < lowest){
				_expandedOnAttribute = key;
				lowest = curInfo_A;
				bestA = possible;
			}
		}
		_possibleExpansions = null;
		return bestA;
	}

	private Map<Double, DecisionNode> expandOn(Integer attr) {
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
		
		Map<Double, DecisionNode> possibleChildren = new HashMap<Double, DecisionNode>();
		
		for(Double key: attributeToInstances.keySet()){
			DecisionNode possibleChild = new DecisionNode();
			possibleChild.setTitle(key);
			
			for(int i = 0; i < _remainingAttributes.size(); i++){
				if(_remainingAttributes.get(i) != attr){
					possibleChild.addAttribute(_remainingAttributes.get(i));
//					System.out.println(_remainingAttributes.get(i));
				}
			}
			
			List<Double[]> curListOfInstances = attributeToInstances.get(key);
			List<Double> curListOfOutputs = attributeToOutputs.get(key);
			
			for(int i = 0; i < curListOfInstances.size(); i++){
				Double[] instance = curListOfInstances.get(i);
				Double output = curListOfOutputs.get(i);
				
				possibleChild.addInstance(instance, output);
//				System.out.println(": " + labels.get(i, 0));
			}
			
			possibleChild.setOutputClasses(this._outputClasses);
			double infoOfPossibleChild = possibleChild.calculateInformation();
			//System.out.println(infoOfPossibleChild);
			
			possibleChildren.put(possibleChild.getTitle(), possibleChild);
		}
		
		return possibleChildren;
	}

	public void addAttribute(Integer attribute){
		_remainingAttributes.add(attribute);
	}
	
	public void addInstance(Double[] instance, Double output){
		_instanceInputs.add(instance);
		_instanceOutputs.add(output);
		_countInstances++;
		
		int outputCount = 0;
		if(_majorityTraker.containsKey(output)){
			outputCount = _majorityTraker.get(output);
		}
		outputCount++;
		_majorityTraker.put(output, outputCount);
		if(_majorityTraker.keySet().size() > 1) _isPure = false;
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
	
	public double getInfo(){
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

	public int getCountInstances() {
		return _countInstances;
	}

	public Double getTitle() {
		return _title;
	}

	public void setTitle(Double _title) {
		this._title = _title;
	}

	public double inferOutput(double[] features) {
		if(_isLeaf){
			return _nodeOutput;
		}else{
			
		}
		return 0;
	}
	
	
}
