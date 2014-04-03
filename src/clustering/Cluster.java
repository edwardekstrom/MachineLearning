package clustering;

import java.util.*;

public class Cluster {
	public Double _distance;
	public String _name;
	public Cluster _parent;
	public List<Cluster> _children = new ArrayList<Cluster>();
	
	public Cluster(){
		_distance = new Double(0);
	}
	
	public double totalDist(){
		double dist;
		if(_distance == null) dist = 0;
		else dist = _distance;
		if(_children.size() > 0){
			dist += _children.get(0).totalDist();
		}
		return dist;
	}
}
