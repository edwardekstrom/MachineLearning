package clustering;

public class Pair implements Comparable<Pair>{
	public Cluster _left;
	public Cluster _right;
	public Double _linkDist;
	@Override
	public int compareTo(Pair pair) {
		int toReturn;
		if(pair == null || pair._linkDist == null)
			toReturn = -1;
		else if( this._linkDist == null)
			toReturn = 1;
		else
			toReturn = this._linkDist.compareTo(pair._linkDist);
		return toReturn;
	}
	
	public Cluster joinClusters(){
		Cluster newClust = new Cluster();
		newClust._name = _left._name + ", " + _right._name.substring(10);
		newClust._distance = _linkDist;
		newClust._children.add(_left);
		newClust._children.add(_right);
		_left._parent = newClust;
		_right._parent = newClust;
		return newClust;
	}
}
