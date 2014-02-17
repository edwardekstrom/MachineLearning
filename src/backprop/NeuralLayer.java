package backprop;

public interface NeuralLayer {
	
	public NeuralNode[] getNodes();
	
	public void connectToLayer(NeuralLayerI nl);
	
}
