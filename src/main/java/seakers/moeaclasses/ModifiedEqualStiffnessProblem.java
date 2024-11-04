package seakers.moeaclasses;

import com.mathworks.engine.MatlabEngine;
import org.moeaframework.core.Solution;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;

public class ModifiedEqualStiffnessProblem extends ConstantRadiusTrussProblem2 {

    /**
     * Modified Equal Stiffness problem class with a feasible stiffness ratio violation delta value incorporated
     */

    private double feasibleStiffnessDelta;

    public ModifiedEqualStiffnessProblem(String savePath, int modelSelection, int numVariables, int numHeurObjectives, int numHeurConstraints, double rad, double sideLength, double E, double sideNodeNum, double nucFac, double targetCRatio, MatlabEngine eng, boolean[][] constrainHeuristics, double feasibleStiffnessDelta) {
        super(savePath, modelSelection, numVariables, numHeurObjectives, numHeurConstraints, rad, sideLength, E, sideNodeNum, nucFac, targetCRatio, eng, constrainHeuristics);
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
