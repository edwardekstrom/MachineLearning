package decision;

import java.util.*;

public class DecisionNode {
	public DecisionTree _decidionTree;
	
	private List<Integer> _remainingAttributes;
	private List<Double[]> _instanceInputs;
	private List<Double> _instanceOutputs;
	private int _countInstances;
	private Set<Double> _outputClasses;
	private Map<Double, Integer> _majorityTraker;
	private double _information;
	private boolean _isLeaf;
	
	private boolean _isPure;
	
	private DecisionNode _parentNode;
	
	private Integer _expandedOnAttribute;
	
	private Map<Double, DecisionNode> _actualChildren;
	
	private Map<Integer, Map<Double, DecisionNode> > _possibleExpansions;
	
	private Double _nodeOutput;
	
	private Double _title;
	
	private Integer _attributeTitle;
	
	
	
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
		
		_parentNode = null;
		
		_expandedOnAttribute = null;
		_nodeOutput = null;
		
		setAttributeTitle(null);
	}
	
	public void expandTree(Double parentOutput){
		if(_countInstances == 0){
//			for(int i = 0 ; i < _countInstances ; i++){
//				for(int j = 0 ; j < _instanceInputs.get(i).length ; j++){
//					System.out.print(_instanceInputs.get(i)[j] + " ");
//				}
//				System.out.println(": " + _instanceOutputs.get(i));
//			}
//			System.out.print("\n\n\n");
			
			_isLeaf = true;
			_decidionTree.addLeaf(this);
			_nodeOutput = parentOutput;
			return;
		}else if(_isPure){
//			for(int i = 0 ; i < _countInstances ; i++){
//				for(int j = 0 ; j < _instanceInputs.get(i).length ; j++){
//					System.out.print(_instanceInputs.get(i)[j] + " ");
//				}
//				System.out.println(": " + _instanceOutputs.get(i));
//			}
//			System.out.print("\n\n\n");
			
			_isLeaf = true;
			_decidionTree.addLeaf(this);
			_nodeOutput = getMajority();
			return;
		}else if(_remainingAttributes.size() == 0){
//			for(int i = 0 ; i < _countInstances ; i++){
//				for(int j = 0 ; j < _instanceInputs.get(i).length ; j++){
//					System.out.print(_instanceInputs.get(i)[j] + " ");
//				}
//				System.out.println(": " + _instanceOutputs.get(i));
//			}
//			System.out.print("\n\n\n");
			
			_isLeaf = true;
			_decidionTree.addLeaf(this);
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
				double childInstances = child.getCountInstances();
				double childInfo = child.getInfo();
				curInfo_A += (childInstances / _countInstances) * childInfo;
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
			possibleChild._parentNode = this;
			possibleChild._decidionTree = this._decidionTree;
			possibleChild.setTitle(key);
			possibleChild.setAttributeTitle(attr);
			
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
	
	public String getTitleString(){
		Integer titleString = new Integer(_title.intValue());
		return titleString.toString();
	}

	public void setTitle(Double _title) {
		this._title = _title;
	}

	public double inferOutput(double[] features) {
		double toReturn = 0d;
		if(_isLeaf){
			toReturn = _nodeOutput;
		}else if(_actualChildren.containsKey(features[_expandedOnAttribute])){
			toReturn = _actualChildren.get(features[_expandedOnAttribute]).inferOutput(features);
		}else{
			toReturn = _nodeOutput;
		}
		return toReturn;
	}
	
	public double inferOutput(Double[] features) {
		double toReturn = 0d;
		if(_isLeaf){
			toReturn = _nodeOutput;
		}else if(_actualChildren.containsKey(features[_expandedOnAttribute])){
			toReturn = _actualChildren.get(features[_expandedOnAttribute]).inferOutput(features);
		}else{
			toReturn = _nodeOutput;
		}
		return toReturn;
	}

	public String makeDot() {
		String toReturn = "";
		String ExpandOn = "Expand On: " + _expandedOnAttribute;
		String Count = "Count: " + _countInstances;
		String Title = "Attribute: " + getAttributeTitle() + "=" + getTitle();
		String values = "";
		for(Double output: _majorityTraker.keySet()){
			values += output + ":" + _majorityTraker.get(output) + "   ";
		}
		toReturn += this.toString().replace(".", "").replace("@", "")
				+ "[label=\"" 
				+ Title + "\n"
				+ Count+ "\n"
				+ values + "\n"
				+ ExpandOn
				+ "\"];";
		if(_isLeaf){
		}else{

			for(Double key: _actualChildren.keySet()){
				toReturn += this.toString().replace(".", "").replace("@", "");
				toReturn += " -> ";
				toReturn += _actualChildren.get(key).toString().replace(".", "").replace("@", "");
				toReturn += ";\n";
				toReturn += _actualChildren.get(key).makeDot();
			} 
		}
		return toReturn;
	}
	
	public String getExpandedOn(){
		if(_expandedOnAttribute == null){
			return "";
		}else{
			return _expandedOnAttribute.toString();
		}
	}

	public Integer getAttributeTitle() {
		return _attributeTitle;
	}

	public void setAttributeTitle(Integer _attributeTitle) {
		this._attributeTitle = _attributeTitle;
	}

	public void pruneUp(List<Double[]> validationInstanceInputs,
			List<Double> validationInstanceOutputs) {
		double curBest = getValidationAccuracy(validationInstanceInputs, validationInstanceOutputs);
		while(_parentNode != null){
			if(_parentNode._isLeaf) break;
			_parentNode._isLeaf = true;
			double accur = getValidationAccuracy(validationInstanceInputs, validationInstanceOutputs);
			if(accur < curBest){
				_parentNode._isLeaf = false;
				break;
			}else{
				_parentNode._expandedOnAttribute = null;
				curBest = accur;
			}
		}
		
	}
	
	public double getValidationAccuracy(List<Double[]> validationInstanceInputs,
			List<Double> validationInstanceOutputs){
		double total = validationInstanceInputs.size();
		double correct = 0;
		for(int i = 0 ; i < validationInstanceInputs.size() ; i++){
			Double[] input = validationInstanceInputs.get(i);
			Double output = validationInstanceOutputs.get(i);
			if (_decidionTree._rootNode.inferOutput(input) == output){
				correct++;
			}
		}
		return correct/total;
	}

	public void removePrunedBranches() {
		if(_isLeaf){
			_actualChildren = new HashMap<Double, DecisionNode>();
		}else{
			for (double key: _actualChildren.keySet()){
				_actualChildren.get(key).removePrunedBranches();
			}
		}
	}
	
	public int countNodes(){
		if(_isLeaf){
			return 1;
		}else{
			int childrenTotal = 0;
			for (double key: _actualChildren.keySet()){
				childrenTotal += _actualChildren.get(key).countNodes();
			}
			return 1 + childrenTotal;
		}
	}
	
	public int maxDepth (DecisionNode r) {
	    int depth = 0;
	    Stack<DecisionNode> wq = new Stack<DecisionNode>();
	    Stack<DecisionNode> path = new Stack<DecisionNode>();

	    wq.push (r);
	    while (!wq.empty()) {
	        r = wq.peek();
	        if (!path.empty() && r == path.peek()) {
	            if (path.size() > depth)
	                depth = path.size();
	            path.pop();
	            wq.pop();
	        } else {
	            path.push(r);
	            for(double key: _actualChildren.keySet()){
	            	DecisionNode child = _actualChildren.get(key);
	            	wq.push(child);
	            }
	        }
	    }

	    return depth;
	}
	
}
