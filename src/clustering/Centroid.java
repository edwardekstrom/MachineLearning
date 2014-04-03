package clustering;

public class Centroid implements Comparable<Centroid>{
	public double[] array;
	public String name;
	public Centroid(){
		array = null;
		name = null;
	}
	@Override
	public int compareTo(Centroid arg0) {
		return name.compareTo(arg0.name);
	}
}
