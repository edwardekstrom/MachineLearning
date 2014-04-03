package clustering;

import java.util.Collection;

public class SingleLink implements LinkInterface {

	@Override
	public Double calculateDistance(Collection<Double> distances) {
		double min = Double.MAX_VALUE;
		for(Double distance: distances){
			if(distance < min)
				min = distance;
		}
		return min;
	}

}
