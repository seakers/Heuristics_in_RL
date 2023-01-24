package seakers.eossrl.actions.assignment;

import seakers.vassarexecheur.search.operators.assigning.RepairPackingEfficiencyAdditionAssigning;
import seakers.vassarheur.ResourcePool;
import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.problems.Assigning.ArchitectureEvaluator;

public class RepairPackingEfficiency extends RepairPackingEfficiencyAdditionAssigning {

    private final double threshold; // Minimum packing efficiency to satisfy heuristic
    private int numInstruments; // number of instruments to add per non-satisfying satellite
    private int numSatellites; // number of non-satisfying satellites to add instruments to
    private AssigningProblem problem; // problem class for architecture evaluation
    private final ResourcePool resourcePool; // used for architecture evaluation
    private final BaseParams params; // used for architecture evaluation
    private final ArchitectureEvaluator evaluator; // used for architecture evaluation

    public RepairPackingEfficiency(double threshold, int numInstruments, int numSatellites, AssigningProblem problem, ResourcePool resourcePool, BaseParams params, ArchitectureEvaluator evaluator) {
        super(threshold, numInstruments, numSatellites, params, problem, resourcePool, evaluator);
        this.threshold = threshold;
        this.numInstruments = numInstruments;
        this.numSatellites = numSatellites;
        this.problem = problem;
        this.resourcePool = resourcePool;
        this.params = params;
        this.evaluator = evaluator;
    }

    // All parent methods can be used, below are modified versions of the methods to override them if needed

    /*
     * Methods: 
     *  1. Solution[] evolve(Solution[] sols)
     *  2. int getArity()
     */
    
}