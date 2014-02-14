package backprop;

public class Backprop {
	private Integer[] _structure;
	private int _numInputNodes;
	private int _numOutputNodes;
	
	public Backprop(Integer[] structure){
		_structure = structure;
		_numInputNodes = _structure[0];
		_numOutputNodes = _structure[_structure.length];
	}
}
