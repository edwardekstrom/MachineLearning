package backprop;

public interface NeuralNode {
	
	public double getOutput();
	
	public void setOutput(double out);
	
	public double getNet();
	
	public void setNet(double net);
	
	public double getWeightTo(NeuralNode node);
	
	public void setWeigthTo(NeuralNode node, double newWeight);
	
	public void setToUpdate(NeuralNode node, double updateValue);
	
	public void setDelta(double d);
	
	public double getDelta();
	
	public double getDeltaSum();
	
	public void update();
	
}
