package seakers.moeaclasses;

import com.mathworks.engine.MatlabEngine;
import org.moeaframework.core.Solution;
import seakers.trussaos.problems.ConstantRadiusArteryProblem;

public class ModifiedArteryProblem extends ConstantRadiusArteryProblem {

    /**
     * Modified Artery problem class with a feasible stiffness ratio violation delta value incorporated
     */

    private double feasibleStiffnessDelta;

    public ModifiedArteryProblem(String savePath, int modelSelection, int numVariables, int numHeurObjectives, int numHeurConstraints, double rad, double sideLength, double E, double sideNodeNum, double nucFac, double targetStiffnessRatio, MatlabEngine eng, boolean[][] constrainHeuristics, double feasibleStiffnessDelta) {
        super(savePath, modelSelection, numVariables, numHeurObjectives, numHeurConstraints, rad, sideLength, E, sideNodeNum, nucFac, targetStiffnessRatio, eng, constrainHeuristics);

        this.feasibleStiffnessDelta = feasibleStiffnessDelta;
    }

    @Override
    public void evaluate(Solution sltn) {
        super.evaluate(sltn);

        // Update stiffness ratio violation if violation <= feasible stiffness delta
        double currentStiffnessRatioViolation = (double) sltn.getAttribute("StiffnessRatioViolation");
        if (currentStiffnessRatioViolation <= this.feasibleStiffnessDelta) {
            sltn.setAttribute("StiffnessRatioViolation", 0);
        }
    }
}
