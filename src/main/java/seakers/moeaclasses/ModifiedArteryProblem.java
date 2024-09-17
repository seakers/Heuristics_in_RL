package seakers.moeaclasses;

import com.mathworks.engine.MatlabEngine;
import seakers.trussaos.problems.ConstantRadiusArteryProblem;

public class ModifiedArteryProblem extends ConstantRadiusArteryProblem {

    public ModifiedArteryProblem(String savePath, int modelSelection, int numVariables, int numHeurObjectives, int numHeurConstraints, double targetStiffnessRatio, MatlabEngine eng, boolean[][] constrainHeuristics) {
        super(savePath, modelSelection, numVariables, numHeurObjectives, numHeurConstraints, targetStiffnessRatio, eng, constrainHeuristics);
    }


}
