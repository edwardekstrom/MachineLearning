package clustering;

import java.util.Collection;

public class CompleteLink implements LinkInterface {

	@Override
	public Double calculateDistance(Collection<Double> distances) {
		double max = -1;
		for(Double distance: distances){
			if (distance>max)
				max = distance;
		}
		return max;
	}

}
