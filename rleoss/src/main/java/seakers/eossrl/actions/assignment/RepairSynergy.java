package seakers.eossrl.actions.assignment;

import seakers.vassarexecheur.search.operators.assigning.RepairSynergyAdditionAssigning;
import java.util.HashMap;
import seakers.vassarheur.ResourcePool;
import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.problems.Assigning.ArchitectureEvaluator;

public class RepairSynergy extends RepairSynergyAdditionAssigning {

    private int numInstruments; // Number of instruments to add
    private HashMap<String, String[]> synergyMap; // mapping of instruments to their synergistic instrument sets
    private AssigningProblem problem; // problem class for architecture evaluation
    private final ResourcePool resourcePool; // used for architecture evaluation
    private final BaseParams params; // used for architecture evaluation
    private final ArchitectureEvaluator evaluator; // used for architecture evaluation

    public RepairSynergy(int numInstruments, HashMap<String, String[]> synergyMap, AssigningProblem problem, ResourcePool resourcePool, BaseParams params, ArchitectureEvaluator evaluator) {
        super(numInstruments, resourcePool, evaluator, params, problem, synergyMap);
        this.numInstruments = numInstruments;
        this.synergyMap = synergyMap;
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