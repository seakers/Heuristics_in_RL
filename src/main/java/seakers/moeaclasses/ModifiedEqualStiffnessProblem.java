package seakers.moeaclasses;

import com.mathworks.engine.MatlabEngine;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;

public class ModifiedEqualStiffnessProblem extends ConstantRadiusTrussProblem2 {
    public ModifiedEqualStiffnessProblem(String savePath, int modelSelection, int numVariables, int numHeurObjectives, int numHeurConstraints, double targetCRatio, MatlabEngine eng, boolean[][] constrainHeuristics) {
        super(savePath, modelSelection, numVariables, numHeurObjectives, numHeurConstraints, targetCRatio, eng, constrainHeuristics);
    }


}
