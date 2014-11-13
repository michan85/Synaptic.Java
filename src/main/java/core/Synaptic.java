package core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import core.Synaptic.Neuron.ConnectedResult;
import core.Synaptic.Neuron.Squasher;



public class Synaptic {

	

	public static class Neuron {

		/*
function Neuron() {
  this.ID = Neuron.uid();
  this.label = null;
  this.connections = {
    inputs: {},
    projected: {},
    gated: {}
  };
  this.error = {
    responsibility: 0,
    projected: 0,
    gated: 0
  };
  this.trace = {
    elegibility: {},
    extended: {},
    influences: {}
  };
  this.state = 0;
  this.old = 0;
  this.activation = 0;
  this.selfconnection = new Neuron.connection(this, this, 0); // weight = 0 -> not connected
  this.squash = Neuron.squash.LOGISTIC;
  this.neighboors = {};
  this.bias = Math.random() * .2 - .1;
}
		 */
		int ID = Neuron.uid();
		String label = null;
		NeuronConnection connections = new NeuronConnection();

		public static class NeuronConnection {
			Map<Integer, Connection> inputs = new HashMap<Integer, Connection>();
			Map<Integer, Connection> projected = new HashMap<Integer, Connection>();
			Map<Integer, Connection> gated = new HashMap<Integer, Connection>();
		}
		Error error = new Error();

		public static class Error {
			double responsibility = 0;
			double projected = 0;
			double gated = 0;
		}

		Trace trace = new Trace();

		public static class Trace {
			Map<Integer, Double> elegibility = new HashMap<Integer, Double>();
			Map<Integer, Map<Integer, Double>> extended = new HashMap<Integer, Map<Integer, Double>>();
			Map<Integer, List<Connection>> influences = new HashMap<Integer, List<Connection>>();
		}

		double state = 0;
		double old = 0;
		double activation = 0;
		double derivative = 0;
		Connection selfconnection;
		Squasher squash = Neuron.Squash.LOGISTIC;
		Map<Integer, Neuron> neighboors = new HashMap<Integer, Synaptic.Neuron>();

		double bias = Math.random() * 0.2 - 0.1;

		public Neuron() {
			selfconnection = new Neuron.Connection(this, this, 0); // weight = 0 -> not connected
		}

		// activate the neuron
		public double activate() {
			return activate(null);
		}

		public double activate(Double input) {
	
			if (input == null) {
				this.activation = input;
				this.derivative = 0;
				this.bias = 0;
				return this.activation;
			}
			this.old = this.state;

			// eq. 15
			this.state = this.selfconnection.gain * this.selfconnection.weight
					* this.state + this.bias;

			for (Connection i : connections.inputs.values()) {
				this.state += i.from.activation * i.weight * i.gain;
			}
			// eq. 16
			this.activation = this.squash(this.state);
			// f'(s)
			this.derivative = this.squash(this.state, true);

			// update traces
			for (Connection inp : connections.inputs.values()) {
				// elegibility trace - Eq. 17
				trace.elegibility.put(inp.ID,
						this.selfconnection.gain * this.selfconnection.weight
								* trace.elegibility.get(inp.ID) + inp.gain
								* inp.from.activation);

				for (Integer id : this.trace.extended.keySet()) {
					Map<Integer, Double> xtrace = trace.extended.get(id);

					Neuron neuron = this.neighboors.get(id);

					// if gated neuron's selfconnection is gated by this unit,
					// the influence keeps track of the neuron's old state
					double influence = neuron.selfconnection.gater == this ? neuron.old: 0;

					// index runs over all the incoming connections to the gated
					// neuron that are gated by this unit
					for (Connection incoming : trace.influences.get(neuron.ID)) { 
						// captures the effect that has an input connection to this unit,
						// on a neuron that is gated by this unit
						influence += incoming.weight * incoming.from.activation;
					}

					// eq. 18
					xtrace.put(
							inp.ID,
							neuron.selfconnection.gain
									* neuron.selfconnection.weight
									* xtrace.get(inp.ID) + this.derivative
									* this.trace.elegibility.get(inp.ID)
									* influence);
				}
			}

			// update gated connection's gains
			for (Connection connection : this.connections.gated.values()) {
				connection.gain = this.activation;
			}

			return this.activation;
		}

		

		// back-propagate the error
		public void propagate(double rate, Double target) {
			
			
			// error accumulator
			double error = 0;

			// output neurons get their error from the enviroment
			if (target != null)
				error = target - this.activation;

			// error responsibilities from all the connections projected from
			// this neuron
			for (Connection connection : this.connections.projected.values()) {
				Neuron neuron = connection.to;
				// Eq. 21
				error += neuron.error.responsibility * connection.gain * connection.weight;
			}

			// projected error responsibility
			this.error.projected = this.derivative * error;

			error = 0;
			// error responsibilities from all the connections gated by this neuron
			for (Integer id : this.trace.extended.keySet()) {
				Neuron neuron = this.neighboors.get(id); // gated neuron
				double influence = neuron.selfconnection.gater == this ? neuron.old
						: 0; // if gated neuron's selfconnection is gated by
								// this neuron

				// index runs over all the connections to the gated neuron that
				// are gated by this neuron
				for (Connection input : this.trace.influences.get(id)) { 
					// capturesthe effect that the input connection of this neuron have,
					// on a neuron which its input/s is/are gated by this neuron
					influence += input.weight * input.from.activation;
				}
				// eq. 22
				error += neuron.error.responsibility * influence;
			}

			// gated error responsibility
			this.error.gated = this.derivative * error;

			// error responsibility - Eq. 23
			this.error.responsibility = this.error.projected + this.error.gated;

			// learning rate
			if (rate <= 0)
				rate = 0.1;

			// adjust all the neuron's incoming connections
			for (Connection input : this.connections.inputs.values()) {

				// Eq. 24
				double gradient = this.error.projected
						* this.trace.elegibility.get(input.ID);
				for (Integer id : this.trace.extended.keySet()) {
					Neuron neuron = this.neighboors.get(id);
					gradient += neuron.error.responsibility
							* this.trace.extended.get(neuron.ID).get(input.ID);
				}
				input.weight += rate * gradient; // adjust weights - aka learn
			}

			// adjust bias
			this.bias += rate * this.error.responsibility;
		}

		public Connection project(Neuron neuron, Double weight) {
			// self-connection
			if (neuron == this) {
				this.selfconnection.weight = 1;
				return this.selfconnection;
			}

			// check if connection already exists
			ConnectedResult connected = this.connected(neuron);
			if (connected.type != ConnectionType.None) {
				// update connection
				if (weight != null)
					connected.connection.weight = weight;
				// return existing connection
				return connected.connection;
			}

			// create a new connection
			Connection connection = new Connection(this, neuron, weight);

			// reference all the connections and traces
			this.connections.projected.put(connection.ID, connection);
			this.neighboors.put(neuron.ID, neuron);
			neuron.connections.inputs.put(connection.ID, connection);
			neuron.trace.elegibility.put(connection.ID, 0d);

			for (Map<Integer, Double> trace : neuron.trace.extended.values()) {
				trace.put(connection.ID, 0d);
			}

			return connection;
		}

		public void gate(Connection connection) {
			// add connection to gated list
			this.connections.gated.put(connection.ID, connection);

			Neuron neuron = connection.to;
			if (this.trace.extended.containsKey(neuron.ID)) {
				// extended trace
				this.neighboors.put(neuron.ID, neuron);
				Map<Integer, Double> xtrace = new HashMap<Integer, Double>();
				trace.extended.put(neuron.ID, xtrace);
				for (Connection input : this.connections.inputs.values()) {
					xtrace.put(input.ID, 0d);
				}
			}

			// keep track
			if (this.trace.influences.containsKey(neuron.ID))
				this.trace.influences.get(neuron.ID).add(connection);
			else {
				ArrayList<Connection> conn = new ArrayList<Synaptic.Neuron.Connection>();
				conn.add(connection);
				this.trace.influences.put(neuron.ID, conn);
			}

			// set gater
			connection.gater = this;
		}

		// returns true or false whether the neuron is self-connected or not
		public boolean selfconnected() {
			return this.selfconnection.weight != 0;
		}

		public static class ConnectedResult {
			ConnectionType type = ConnectionType.None;
			Connection connection;
		}

		public static enum ConnectionType {
			Self, Input, Gated, Projected, None
		}

		// returns true or false whether the neuron is connected to another neuron (parameter)
		public ConnectedResult connected(Neuron neuron) {
			ConnectedResult result = new ConnectedResult();

			if (this == neuron) {
				if (this.selfconnected()) {
					result.type = ConnectionType.Self;
					result.connection = this.selfconnection;
					return result;
				} else
					return result;
			}

			for (Connection connection : this.connections.inputs.values()) {

				if (connection.to == neuron || connection.from == neuron) {
					result.type = ConnectionType.Input;
					result.connection = connection;
					return result;
				}
			}

			for (Connection connection : this.connections.gated.values()) {

				if (connection.to == neuron || connection.from == neuron) {
					result.type = ConnectionType.Gated;
					result.connection = connection;
					return result;
				}
			}

			for (Connection connection : this.connections.projected.values()) {

				if (connection.to == neuron || connection.from == neuron) {
					result.type = ConnectionType.Projected;
					result.connection = connection;
					return result;
				}
			}

			return result;
		}

		// clears all the traces (the neuron forgets it's context, but the
		// connections remain intact)
		public void clear() {

			for (Integer trace : this.trace.elegibility.keySet())
				this.trace.elegibility.put(trace, 0d);

			for (Map<Integer, Double> trace : this.trace.extended.values())
				for (Integer extended : trace.keySet())
					trace.put(extended, 0d);

			this.error.responsibility = this.error.projected = this.error.gated = 0;
		}

		// all the connections are randomized and the traces are cleared
		public void reset() {
			this.clear();

			// for (var type : this.connection)
			for (Connection connection : this.connections.inputs.values())
				connection.weight = Math.random() * .2 - .1;
			for (Connection connection : this.connections.gated.values())
				connection.weight = Math.random() * .2 - .1;
			for (Connection connection : this.connections.projected.values())
				connection.weight = Math.random() * .2 - .1;
			this.bias = Math.random() * .2 - .1;

			this.old = this.state = this.activation = 0;
		}

		// represents a connection between two neurons
		public static class Connection {

			int ID = 0;
			Neuron from;
			Neuron to;
			double weight = 0;
			double gain = 0;
			Object gater = null;

			public Connection(Neuron from, Neuron to, double weight) {
				if (from == null || to == null)
					throw new RuntimeException(
							"Connection Error: Invalid neurons");

				this.ID = Neuron.Connection.uid();
				this.from = from;
				this.to = to;
				this.weight = weight == 0 ? Math.random() * .2 - .1 : weight;
				this.gain = 1;
				this.gater = null;
			}

			public static int uid() {
				return connections++;
			}

			static int connections = 0;
		}// end connection

		static int neurons = 0;

		public static int uid() {
			return neurons++;
		}
		
		public double squash(double x) {
			return this.squash.squash(x, false);
		}

		public double squash(double x, boolean d) {
			return this.squash.squash(x, d);
		}

		public static class NeuronQuatity {
			public int neurons = 0;
			public int connections = 0;
		}

		public static NeuronQuatity quantity() {
			NeuronQuatity q = new NeuronQuatity();
			q.neurons = neurons;
			q.connections = Connection.connections;
			return q;
		}

		// squashing functions
		public static interface Squasher {
			public double squash(double x, boolean derivate);
		}

		public static class Squash {
			public static final Squasher LOGISTIC = new Squasher() {

				public double squash(double x, boolean derivate) {
					if (!derivate)
						return 1 / (1 + Math.exp(-x));
					double fx = Neuron.Squash.LOGISTIC.squash(x, false);
					return fx * (1 - fx);
				}
			};

			public static final Squasher TANH = new Squasher() {

				public double squash(double x, boolean derivate) {
					if (derivate)
						return 1 - Math.pow(
								Neuron.Squash.TANH.squash(x, false), 2);
					double eP = Math.exp(x);
					double eN = 1 / eP;
					return (eP - eN) / (eP + eN);
				}
			};

			public static final Squasher IDENTITY = new Squasher() {

				public double squash(double x, boolean derivate) {
					return derivate ? 1 : x;
				}
			};

			public static final Squasher HLIM = new Squasher() {

				public double squash(double x, boolean derivate) {
					return derivate ? 1 : (x > 0 ? 1 : 0);
				}
			};

		}
	}// end Neuron

	
	/*******************************************************************************************
	 * LAYER
	 *******************************************************************************************/

	public static class Layer {
		public Layer(int size) {
			this(size, null);
		}

		public Layer(int size, String label) {
			this.size = size;
			this.label = label;
			while (size-- > 0) {
				Neuron neuron = new Neuron();
				this.list.add(neuron);
			}
		}

		int size = 0;
		List<Neuron> list = new ArrayList<Synaptic.Neuron>();
		String label = null;

		// activates all the neurons the layer
		public List<Double> activate() {
			return activate(null);
		}

		public List<Double> activate(List<Double> input) {

			List<Double> activations = new ArrayList<Double>();

			if (input != null) {
				if (input.size() != this.size)
					throw new RuntimeException(
							"INPUT size and LAYER size must be the same to activate!");

				for (int id = 0; id < list.size(); id++) {
					Neuron neuron = this.list.get(id);
					double activation = neuron.activate(input.get(id));
					activations.add(activation);
				}
			} else {
				for (Neuron neuron : this.list) {
					double activation = neuron.activate();
					activations.add(activation);
				}
			}
			return activations;
		}

		// propagates the error on all the neurons of the layer
		public void propagate(double rate, List<Double> target) {

			if (target != null) {
				if (target.size() != this.size)
					throw new RuntimeException(
							"TARGET size and LAYER size must be the same to propagate!");

				for (int id = this.list.size() - 1; id >= 0; id--) {
					Neuron neuron = this.list.get(id);
					neuron.propagate(rate, target.get(id));
				}
			} else {
				for (int id = this.list.size() - 1; id >= 0; id--) {
					Neuron neuron = this.list.get(id);
					neuron.propagate(rate, null);
				}
			}
		}

		// projects a connection from this layer to another one
		public Layer.Connection project(Network layer, ConnectionType type,
				List<Double> weights) {
			return project(layer.input, type, weights);
		}

		public Layer.Connection project(Layer layer) {
			return project(layer, null, null);
		}

		public Layer.Connection project(Layer layer, ConnectionType type) {
			return project(layer, type, null);
		}

		public Layer.Connection project(Layer layer, ConnectionType type,
				List<Double> weights) {

			if (connected(layer) != null)
				return new Layer.Connection(this, layer, type, weights);
			return null;// TODO|: this isnt the same..

		}

		// gates a connection between two layers
		public void gate(Layer.Connection connection, Layer.GateType type) {

			if (type == Layer.GateType.INPUT) {
				if (connection.to.size != this.size)
					throw new RuntimeException("GATER layer and CONNECTION.TO layer must be the same size  order to gate!");

				for (int id = 0; id < connection.to.list.size(); id++) {
					Neuron neuron = connection.to.list.get(id);
					Neuron gater = this.list.get(id);
					for (core.Synaptic.Neuron.Connection gated : neuron.connections.inputs.values()) {
						if (connection.connections.containsKey(gated.ID))
							gater.gate(gated);
					}
				}
			} else if (type == Layer.GateType.OUTPUT) {
				if (connection.from.size != this.size)
					throw new RuntimeException("GATER layer and CONNECTION.FROM layer must be the same size  order to gate!");

				for (int id = 0; id < connection.from.list.size(); id++) {
					Neuron neuron = connection.from.list.get(id);
					Neuron gater = this.list.get(id);
					for (core.Synaptic.Neuron.Connection gated : neuron.connections.projected.values()) {
						if (connection.connections.containsKey(gated.ID))
							gater.gate(gated);
					}
				}
			} else if (type == Layer.GateType.ONE_TO_ONE) {
				if (connection.size != this.size)
					throw new RuntimeException("The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!");
				for (int id = 0; id < connection.list.size(); id++) {
					
					Neuron gater = this.list.get(id);
					Neuron gated = connection.list.get(id);
					gater.gate(gated);
				}
			}
		}

		// true or false whether the whole layer is self-connected or not
		public boolean selfconnected() {
			for (Neuron neuron : this.list) {
				if (!neuron.selfconnected())
					return false;
			}
			return true;
		}

		// true of false whether the layer is connected to another layer
		// (parameter) or not
		public Layer.ConnectionType connected(Layer layer) {
			// Check if ALL to ALL connection
			int connections = 0;
			for (Neuron from : this.list) {
				for (Neuron to : layer.list) {
					// TODO: check ConnectionType enums, there is one in Neuron&& Layer...
					ConnectedResult connected = from.connected(to);
					if (connected.type == Neuron.ConnectionType.Projected)
						connections++;
				}
			}

			if (connections == this.size * layer.size)
				return Layer.ConnectionType.ALL_TO_ALL;

			// Check if ONE to ONE connection
			connections = 0;
			for (int neuron = 0; neuron < this.list.size(); neuron++) {
				Neuron from = this.list.get(neuron);
				Neuron to = layer.list.get(neuron);
				ConnectedResult connected = from.connected(to);
				if (connected.type == Neuron.ConnectionType.Projected)
					connections++;
			}
			if (connections == this.size)
				return Layer.ConnectionType.ONE_TO_ONE;
			return null;
		}

		// clears all the neuorns the layer
		public void clear() {
			for (Neuron neuron : this.list) {
				neuron.clear();
			}
		}

		// resets all the neurons the layer
		public void reset() {
			for (Neuron neuron : this.list) {
				neuron.reset();
			}
		}

		// returns all the neurons the layer (array)
		public List<Neuron> neurons() {
			return this.list;
		}

		// adds a neuron to the layer
		public void add(Neuron neuron) {
			if (neuron == null)
				neuron = new Neuron();
			this.list.add(neuron);
			this.size++;
		}

		public Layer set(Options options) {
			if (options == null)
				options = new Options();

			for (Neuron neuron : this.list) {
				if (options.label != null)
					neuron.label = options.label + '_' + neuron.ID;
				if (options.squash != null)
					neuron.squash = options.squash;
				if (options.bias != null)
					neuron.bias = options.bias;
			}
			return this;
		}

		public static class Options {
			public String label;
			public Squasher squash;
			public Double bias;

			public Options bias(double b) {
				bias = b;
				return this;
			}

			public Options squash(Squasher b) {
				squash = b;
				return this;
			}

			public Options label(String b) {
				label = b;
				return this;
			}
		}

		// represents a connection from one layer to another, and keeps track of
		// its weight and gain
		public static class Connection {
			int ID = 0;
			Layer from, to;
			boolean selfconnection = false;
			ConnectionType type;
			Map<Integer, Object> connections;
			List<Neuron> list;
			int size = 0;

			public Connection(Layer fromLayer, Layer toLayer,
					ConnectionType type, List<Double> weights) {
				this.ID = Layer.Connection.uid();
				this.from = fromLayer;
				this.to = toLayer;
				this.selfconnection = toLayer == fromLayer;
				this.type = type;
				this.connections = new HashMap<Integer, Object>();
				this.list = new ArrayList<Neuron>();
				this.size = 0;

				if (type == null) {
					if (fromLayer == toLayer)
						this.type = Layer.ConnectionType.ONE_TO_ONE;
					else
						this.type = Layer.ConnectionType.ALL_TO_ALL;
				}

				if (this.type == Layer.ConnectionType.ALL_TO_ALL) {
					for (Neuron from : this.from.list) {
						for (Neuron to : this.to.list) {
							core.Synaptic.Neuron.Connection connection = from.project(to, weights);
							this.connections.put(connection.ID, connection);
							this.size = this.list.size();
						}
					}
				} else if (this.type == Layer.ConnectionType.ONE_TO_ONE) {

					for (int neuron = 0; neuron < this.from.list.size(); neuron++) {
						Neuron from = this.from.list.get(neuron);
						Neuron to = this.to.list.get(neuron);
						core.Synaptic.Neuron.Connection connection = from.project(to, weights);

						this.connections.put(connection.ID, connection);
						this.size = list.size();
					}
				}
			}

			public static int connectionsIds = 0;

			public static int uid() {
				return connectionsIds++;
			}
		}

		// types of connections
		public static enum ConnectionType {
			ALL_TO_ALL, ONE_TO_ONE
		}

		// types of gates
		public static enum GateType {
			INPUT, OUTPUT, ONE_TO_ONE
		}

	}// end layer


	/*******************************************************************************************
	 * NETWORK
	 *******************************************************************************************/

	public static class Network {

		static int networks = 0;

		public static int uid() {
			return networks++;
		}

		Layer input, output;
		List<Layer> hidden;
		Object optimized = null;

		public Network() {
		}

		public Network(Layer input, List<Layer> hidden, Layer output) {
			// if (typeof layers != 'undefined') {
			// this.optimized = null;
			// }
			if (input != null)
				this.input = input;
			if (hidden != null)
				this.hidden = hidden;
			else
				hidden = new ArrayList<Layer>();

			if (output != null)
				this.output = output;

		}

		// Network.prototype = {

		// feed-forward activation of all the layers to produce an ouput
		public List<Double> activate(List<Double> input) {

			// if (this.optimized == false)
			// {
			// this.layers.input.activate(input);
			this.input.activate(input);
			for (Layer layer : this.hidden)
				layer.activate();
			return this.output.activate();
			// }
			// else
			// {//TODO: figure out optimised, its a boolean and a object...
			// if (this.optimized == null)
			// this.optimize();
			// return this.optimized.activate(input);
			// }
		}

		// back-propagate the error thru the network
		public void propagate(double rate, List<Double> target) {

			// if (this.optimized == false)
			// {
			this.output.propagate(rate, target);
			List<Layer> reverse = Arrays.asList(hidden.toArray(new Layer[hidden.size()]));
			Collections.reverse(reverse);

			for (Layer layer : reverse)
				layer.propagate(rate, null);
			// }
			// else
			// {
			// if (this.optimized == null)
			// this.optimize();
			// this.optimized.propagate(rate, target);
			// }
		}

		// project a connection to another unit (either a network or a layer)
		public Layer.Connection project(Network unit, Layer.ConnectionType type, List<Double> weights) {

			// if (this.optimized)
			// this.optimized.reset();

			// if (unit instanceof Network)
			return this.output.project(unit.input, type, weights);
		}

		public Layer.Connection project(Layer unit, Layer.ConnectionType type,
				List<Double> weights) {

			// if (this.optimized)
			// this.optimized.reset();
			return this.output.project(unit, type, weights);

			// throw
			// "Invalid argument, you can only project connections to LAYERS and NETWORKS!";
		}

		// let this network gate a connection
		public void gate(Layer.Connection connection, Layer.GateType type) {
			// if (this.optimized)
			// this.optimized.reset();
			this.output.gate(connection, type);
		}

		// clear all elegibility traces and extended elegibility traces (the
		// network forgets its context, but not what was trained)
		public void clear() {

			this.restore();

			Layer inputLayer = this.input, outputLayer = this.output;

			inputLayer.clear();
			for (Layer layer : this.hidden) {

				layer.clear();
			}
			outputLayer.clear();

			// if (this.optimized)
			// this.optimized.reset();
		}

		// reset all weights and clear all traces (ends up like a new network)
		public void reset() {

			this.restore();

			Layer inputLayer = this.input, outputLayer = this.output;

			inputLayer.reset();
			for (Layer layer : this.hidden) {
				layer.reset();
			}
			outputLayer.reset();

			// if (this.optimized)
			// this.optimized.reset();
		}


		// restores all the values from the optimized network the their
		// respective objects order to manipulate the network
		public void restore() {
			
		}

		public static class NeuronResult {

			public NeuronResult(Neuron n, String layer) {
				this.neuron = n;
				this.layer = layer;
			}

			public final Neuron neuron;
			public final String layer;
		}

		// returns all the neurons the network
		public List<NeuronResult> neurons() {

			List<NeuronResult> neurons = new ArrayList<NeuronResult>();

			List<Neuron> inputLayer = this.input.neurons(), outputLayer = this.output
					.neurons();

			for (Neuron neuron : inputLayer)
				neurons.add(new NeuronResult(neuron, "input"));

			for (Layer layer : this.hidden) {
				// var hiddenLayer = this.layers.hidden[layer].neurons();
				for (Neuron neuron : layer.neurons())
					neurons.add(new NeuronResult(neuron, layer.label));
			}
			for (Neuron neuron : outputLayer)
				neurons.add(new NeuronResult(neuron, "output"));

			return neurons;
		}

		// returns number of inputs of the network
		public int inputs() {
			return this.input.size;
		}

		// returns number of outputs of hte network
		public int outputs() {
			return output.size;
		}

		// sets the layers of the network
		public void set(Layer input, List<Layer> hidden, Layer output) {

			this.input = input;
			this.hidden = hidden;
			this.output = output;


			// if (this.optimized)
			// this.optimized.reset();
		}

		public void setOptimize(boolean bool) {
			this.restore();
			// if (this.optimized)
			// this.optimized.reset();
			// this.optimized = bool? null : false;
		}

		// returns a json that represents all the neurons and connections of the
		// network
		public String toJSON(boolean ignoreTraces) {

			
			return "TODO";
		}


	}// end network

	/*******************************************************************************************
	 * TRAINER
	 *******************************************************************************************/

	public static class Trainer {

		public static class Options {
			public double rate = .5d;
			public long iterations = 100000;
			public double error = .005d;
		}

		double rate = .5d;
		long iterations = 100000;
		double error = .005d;
		Network network;

		public Trainer(Network network) {
			this(network, null);
		}

		public Trainer(Network network, Trainer.Options options) {
			if (options == null)
				options = new Trainer.Options();

			this.network = network;
			this.rate = options.rate;
			this.iterations = options.iterations;
			this.error = options.error;
		}

		public static class Result {
			public double error;
			public long iterations;
			public long time;
		}

		public static class TrainingSet {
			List<Double> input, target;

			public TrainingSet() {
			}

			public TrainingSet(List<Double> input, List<Double> target) {
				this.input = input;
				this.target = target;
			}
		}

		public static class TrainingOptions extends Options {
			public boolean shuffle;
			public int log = -1;
		}

		// trains any given set to a network
		public Result train(List<TrainingSet> set, TrainingOptions options) {

			double error = 1;
			long iterations = 0;
			List<Double> input, output, target;
			long start = now();

			if (options != null) {

				if (options.iterations > 0)
					this.iterations = options.iterations;
				if (options.error > 0)
					this.error = options.error;
				if (options.rate > 0)
					this.rate = options.rate;
			}

			while (iterations < this.iterations && error > this.error) {
				error = 0;

				for (TrainingSet train : set) {
					input = train.input;
					target = train.target;

					output = this.network.activate(input);
					this.network.propagate(this.rate, target);

					double delta = 0;
					for (int i = 0; i < output.size(); i++)
						delta += Math.pow(target.get(i) - output.get(i), 2);

					error += delta / output.size();
				}

				// check error
				iterations++;
				error /= set.size();

				if (options != null) {

					// TODO: TrainingSet Logging
					// if (options.customLog && options.customLog.every &&
					// iterations %
					// options.customLog.every == 0)
					// options.customLog.do({
					// error: error,
					// iterations: iterations
					// });
					// else
					if (options.log > -1 && iterations % options.log == 0) {
						System.out.println("iterations: " + iterations
								+ " error: " + error);
					}
					;
					if (options.shuffle)
						Collections.shuffle(set);
				}
			}

			Result results = new Result();
			results.error = error;
			results.iterations = iterations;
			results.time = now() - start;

			return results;
		}


		// trains an XOR to the network
		public Result XOR(TrainingOptions options) {

			if (this.network.inputs() != 2 || this.network.outputs() != 1)
				throw new RuntimeException(
						"Error: Incompatible network (2 inputs, 1 output)");

			if (options == null) {
				options = new TrainingOptions();
				options.iterations = 100000;
				options.shuffle = true;
			}
			
			List<TrainingSet> set = new ArrayList<TrainingSet>();
			set.add(new TrainingSet(Arrays.asList(0d, 0d), Arrays.asList(0d)));
			set.add(new TrainingSet(Arrays.asList(1d, 0d), Arrays.asList(1d)));
			set.add(new TrainingSet(Arrays.asList(0d, 1d), Arrays.asList(1d)));
			set.add(new TrainingSet(Arrays.asList(1d, 1d), Arrays.asList(0d)));
			return this.train(set, options);
		}

		public static class DSRTrainingOptions {
			List<Double> targets = Arrays.asList(2d, 4d, 7d, 8d);
			List<Double> distractors = Arrays.asList(3d, 5d, 6d, 9d);
			List<Double> prompts = Arrays.asList(0d, 1d);
			int length = 24;
			double criterion = 0.95;
			long iterations = 100000;
			double rate = .1;
			int log = -1;
			CustomLog customLog = null;
		}

		public class DSRResult extends Result {
			public int success = 0;
		}

		// trains the network to pass a Distracted Sequence Recall test
		public DSRResult DSR(DSRTrainingOptions options) {
			
			List<Double> targets = options.targets;
			List<Double> distractors = options.distractors;
			List<Double> prompts = options.prompts;
			int length = options.length;
			double criterion = options.criterion;
			long iterations = options.iterations;
			double rate = options.rate;
			int log = options.log;
			CustomLog customLog = options.customLog;

			int trial = 0, correct = 0, success = 0, error = 1, symbols = targets
					.size() + distractors.size() + prompts.size();

			Function2<Integer, List<Integer>, Integer> noRepeat = new Function2<Integer, List<Integer>, Integer>() {
				public Integer call(Integer range, List<Integer> avoid) {

					int number = random(range);// Math.random() * range | 0;
					boolean used = false;
					for (int i : avoid)
						if (number == avoid.get(i))
							used = true;
					return used ? call(range, avoid) : number;
				}
			};
			Function2<List<Double>, List<Double>, Boolean> equal = new Function2<List<Double>, List<Double>, Boolean>() {
				public Boolean call(List<Double> prediction, List<Double> output) {
					for (int i = 0; i < prediction.size(); i++)
						if (Math.round(prediction.get(i)) != output.get(i))
							return false;
					return true;
				}
			};

			long start = now();

			while (trial < iterations
					&& (success < criterion || trial % 1000 != 0)) {
				// generate sequence
				List<Double> sequence = new ArrayList<Double>();
				int sequenceLength = length - prompts.size();
				for (int i = 0; i < sequenceLength; i++) {
					int any = random(distractors.size());
					sequence.add(distractors.get(any));
				}
				ArrayList<Integer> indexes = new ArrayList<Integer>(), positions = new ArrayList<Integer>();
				for (int i = 0; i < prompts.size(); i++) {
					indexes.add(random(targets.size()));
					positions.add(noRepeat.call(sequenceLength, positions));
				}
				Collections.sort(positions);
				for (int i = 0; i < prompts.size(); i++) {
					sequence.set(positions.get(i), targets.get(indexes.get(i)));
					sequence.add(prompts.get(i));
				}

				// train sequence
				int targetsCorrect = 0, distractorsCorrect = 0;
				error = 0;
				for (int i = 0; i < length; i++) {
					// generate input from sequence
					ArrayList<Double> input = new ArrayList<Double>(symbols);

					input.set(sequence.get(i), 1d);

					// generate target output
					ArrayList<Double> output = new ArrayList<Double>();

					if (i >= sequenceLength) {
						int index = i - sequenceLength;
						output.set(indexes.get(index), 1d);
					}

					// check result
					List<Double> prediction = this.network.activate(input);

					if (equal.call(prediction, output))
						if (i < sequenceLength)
							distractorsCorrect++;
						else
							targetsCorrect++;
					else {
						this.network.propagate(rate, output);
					}

					double delta = 0;
					for (int j = 0; j < prediction.size(); j++)
						delta += Math.pow(output.get(j) - prediction.get(j), 2);
					error += delta / this.network.outputs();

					if (distractorsCorrect + targetsCorrect == length)
						correct++;
				}

				// calculate error
				if (trial % 1000 == 0)
					correct = 0;
				trial++;
				int divideError = trial % 1000;
				divideError = divideError == 0 ? 1000 : divideError;
				success = correct / divideError;
				error /= length;

				// log
				if (log > 0 && trial % log == 0)
					log("iterations:", trial, " success:", success," correct:", correct, " time:", now() - start," error:", error);
				// if ( customLog.every && trial % customLog.every == 0)
				// customLog.log({
				// iterations: trial,
				// success: success,
				// error: error,
				// time: now() - start,
				// correct: correct
				// });
			}

			DSRResult result = new DSRResult();
			result.error = error;
			result.iterations = iterations;
			result.time = now() - start;
			result.success = success;
			return result;
		}

		public static class ERGPath {
			public String value;
			public ERGNode node;

			public ERGPath() {
			}

			public ERGPath(ERGNode node, String value) {
				this.node = node;
				this.value = value;
			}
		}

		public static class ERGNode {

			public List<ERGPath> paths = new ArrayList<ERGPath>();

			public ERGNode connect(ERGNode node, String value) {
				paths.add(new ERGPath(node, value));
				return this;
			}

			public ERGPath any() {
				if (this.paths.size() == 0)
					return null;
				int index = random(paths.size());// Math.random() *
													// this.paths.size() | 0;
				return this.paths.get(index);
			}

			public ERGPath test(String value) {
				for (int i = 0; i < this.paths.size(); i++)
					if (this.paths.get(i).value.equals(value))
						return this.paths.get(i);
				return null;
			}
		}

		public static class GrammarResult {
			public ERGNode input, output;

			public GrammarResult() {
			}

			public GrammarResult(ERGNode i, ERGNode o) {
				this.input = i;
				this.output = o;
			}
		}

		// train the network to learn an Embeded Reber Grammar
		public ERGResult ERG(TrainingOptions options) {

			if (options == null) {
				options = new TrainingOptions();
				options.iterations = 150000;
				options.error = 0.05;
				options.rate = .1;
				options.log = 500;
			}


			long iterations = options.iterations;
			double criterion = options.error;
			double rate = options.rate;
			int log = options.log;

			

			final Function<GrammarResult> reberGrammar = new Function<GrammarResult>() {
				public GrammarResult call() {
					// build a reber grammar
					ERGNode output = new ERGNode();
					ERGNode n1 = (new ERGNode()).connect(output, "E");
					ERGNode n2 = (new ERGNode()).connect(n1, "S");
					ERGNode n3 = (new ERGNode()).connect(n1, "V").connect(n2,
							"P");
					ERGNode n4 = (new ERGNode()).connect(n2, "X");
					n4.connect(n4, "S");
					ERGNode n5 = (new ERGNode()).connect(n3, "V");
					n5.connect(n5, "T");
					n2.connect(n5, "X");
					ERGNode n6 = (new ERGNode()).connect(n4, "T").connect(n5,
							"P");
					ERGNode input = (new ERGNode()).connect(n6, "B");

					return new GrammarResult(input, output);

				}
			};

			// build an embeded reber grammar

			final Function<GrammarResult> embededReberGrammar = new Function<GrammarResult>() {
				public GrammarResult call() {

					GrammarResult reber1 = reberGrammar.call();
					GrammarResult reber2 = reberGrammar.call();

					ERGNode output = new ERGNode();
					ERGNode n1 = (new ERGNode()).connect(output, "E");
					reber1.output.connect(n1, "T");
					reber2.output.connect(n1, "P");
					ERGNode n2 = (new ERGNode()).connect(reber1.input, "P")
							.connect(reber2.input, "T");
					ERGNode input = (new ERGNode()).connect(n2, "B");

					return new GrammarResult(input, output);

				}
			};

			// generate an ERG sequence
			final Function<String> generate = new Function<String>() {
				public String call() {

					ERGNode node = embededReberGrammar.call().input;
					ERGPath next = node.any();
					String str = "";
					while (next != null) {
						str += next.value;
						next = next.node.any();
					}
					return str;
				}
			};

			// test if a string matches an embeded reber grammar
			final Function1<String, Boolean> test = new Function1<String, Boolean>() {

				public Boolean call(String str) {
					ERGNode node = embededReberGrammar.call().input;
					int i = 0;
					char ch = str.charAt(i);
					while (i < str.length()) {
						ERGPath next = node.test(ch + "");
						if (next != null)
							return false;
						node = next.node;
						ch = str.charAt(++i);
					}
					return true;
				}
			};

			// helper to check if the output and the target vectors match
			final Function2<List<Double>, List<Double>, Boolean> different = new Function2<List<Double>, List<Double>, Boolean>() {

				public Boolean call(List<Double> array1, List<Double> array2) {
					double max1 = 0;
					int i1 = -1;
					double max2 = 0;
					int i2 = -1;
					for (int i = 0; i < array1.size(); i++) {
						if (array1.get(i) > max1) {
							max1 = array1.get(i);
							i1 = i;
						}
						if (array2.get(i) > max2) {
							max2 = array2.get(i);
							i2 = i;
						}
					}

					return i1 != i2;
				}
			};

			int iteration = 0;
			double error = 1;
			Map<String, Integer> table = new HashMap<String, Integer>();
			table.put("B", 0);
			table.put("P", 1);
			table.put("T", 2);
			table.put("X", 3);
			table.put("S", 4);
			table.put("E", 5);
			

			long start = now();
			while (iteration < iterations && error > criterion) {
				int i = 0;
				error = 0;

				// ERG sequence to learn
				String sequence = generate.call();

				// input
				char read = sequence.charAt(i);
				// target
				char predict = sequence.charAt(i + 1);

				// train
				while (i < sequence.length() - 1) {
					List<Double> input = new ArrayList<Double>();
					List<Double> target = new ArrayList<Double>();
					for (int j = 0; j < 6; j++) {
						input.set(j, 0d);
						target.set(j, 0d);
					}
					input.set(table.get(read), 1d);
					target.set(table.get(predict), 1d);

					List<Double> output = this.network.activate(input);

					if (different.call(output, target))
						this.network.propagate(rate, target);

					read = sequence.charAt(++i);
					predict = sequence.charAt(i + 1);

					double delta = 0;
					for (int k = 0; k < output.size(); k++)
						delta += Math.pow(target.get(k) - output.get(k), 2);
					delta /= output.size();

					error += delta;
				}
				error /= sequence.length();
				iteration++;
				if (iteration % log == 0) {
					log("iterations:", iteration, " time:", now() - start,
							" error:", error);
				}
			}

			ERGResult result = new ERGResult();
			result.error = error;
			result.iterations = iterations;
			result.time = now() - start;
			result.test = test;
			result.generate = generate;
			return result;
			
		}

		public static class ERGResult extends Result {
			public Function1<String, Boolean> test;
			public Function<String> generate;
		}
	}// end trainer

	/*******************************************************************************************
	 * ARCHITECT
	 *******************************************************************************************/

	// Colection of useful built- architectures
	public static class Architect {
		public static class Perceptron extends Network {
			Trainer trainer;

			public Perceptron(int... args) {

				// var args = Array.prototype.slice.call(arguments); // convert
				// arguments to Array
				if (args.length < 3)
					throw new RuntimeException("Error: not enough layers (minimum 3) !!");


				int inputs = args[0];
				int outputs = args[args.length - 1];
				List<Integer> layers = new ArrayList<Integer>();
				for (int i = 1; i < args.length - 2; i++)
					layers.add(args[i]);

				Layer input = new Layer(inputs, null);
				List<Layer> hidden = new ArrayList<Layer>();
				Layer output = new Layer(outputs, null);

				Layer previous = input;

				// generate hidden layers
				for (int level = 0; level < layers.size(); level++) {
					int size = layers.get(level);
					Layer layer = new Layer(size, null);
					hidden.add(layer);
					previous.project(layer);
					previous = layer;
				}
				previous.project(output);

				// set layers of the neural network
				this.set(input, hidden, output);

				// trainer for the network
				this.trainer = new Trainer(this);
			}
		}// end perceptron

		// Multilayer Long Short-Term Memory
		public static class LSTM extends Network {
			Trainer trainer;

			public LSTM(int... args) {
				// var args = Array.prototype.slice.call(arguments); // convert
				// arguments to Array
				if (args.length < 3)
					throw new RuntimeException("Error: not enough layers (minimum 3) !!");

				int inputs = args[0];
				int outputs = args[args.length - 1];
				List<Integer> layers = new ArrayList<Integer>();
				for (int i = 1; i < args.length - 2; i++)
					layers.add(args[i]);

				Layer inputLayer = new Layer(inputs, null);
				List<Layer> hiddenLayers = new ArrayList<Layer>();
				Layer outputLayer = new Layer(outputs, null);
				Layer previous = null;

				// generate layers
				for (int layer = 0; layer < layers.size(); layer++) {
					// generate memory blocks (memory cell and respective gates)
					int size = layers.get(layer);

					Layer inputGate = new Layer(size).set(new Layer.Options().bias(1));
					Layer forgetGate = new Layer(size).set(new Layer.Options().bias(1));
					Layer memoryCell = new Layer(size);
					Layer outputGate = new Layer(size).set(new Layer.Options().bias(1));

					hiddenLayers.add(inputGate);
					hiddenLayers.add(forgetGate);
					hiddenLayers.add(memoryCell);
					hiddenLayers.add(outputGate);

					// connections from input layer
					Layer.Connection input = inputLayer.project(memoryCell);
					inputLayer.project(inputGate);
					inputLayer.project(forgetGate);
					inputLayer.project(outputGate);

					// connections from previous memory-block layer to this one
					if (previous != null) {
						Layer.Connection cell = previous.project(memoryCell);
						previous.project(inputGate);
						previous.project(forgetGate);
						previous.project(outputGate);
					}

					// connections from memory cell
					Layer.Connection output = memoryCell.project(outputLayer);

					// self-connection
					Layer.Connection self = memoryCell.project(memoryCell);

					// peepholes
					memoryCell.project(inputGate,
							Layer.ConnectionType.ONE_TO_ONE);
					memoryCell.project(forgetGate,
							Layer.ConnectionType.ONE_TO_ONE);
					memoryCell.project(outputGate,
							Layer.ConnectionType.ONE_TO_ONE);

					// gates
					inputGate.gate(input, Layer.GateType.INPUT);
					forgetGate.gate(self, Layer.GateType.ONE_TO_ONE);
					outputGate.gate(output, Layer.GateType.OUTPUT);
					if (previous != null)
						inputGate.gate(cell, Layer.GateType.INPUT);

					previous = memoryCell;
				}

				// input to output direct connection
				inputLayer.project(outputLayer);

				// set the layers of the neural network
				this.set(inputLayer, hiddenLayers, outputLayer);

				// trainer
				this.trainer = new Trainer(this);
			}
		}// end lstm

		// Liquid State Machine
		public static class Liquid extends Network {
			Trainer trainer;

			public Liquid(int inputs, int hidden, int outputs, int connections,
					int gates) {

				// create layers
				Layer inputLayer = new Layer(inputs);
				Layer hiddenLayer = new Layer(hidden);
				Layer outputLayer = new Layer(outputs);

				// make connections and gates randomly among the neurons
				List<Neuron> neurons = hiddenLayer.neurons();
				List<Neuron.Connection> connectionList = new ArrayList<Neuron.Connection>();

				for (int i = 0; i < connections; i++) {
					// connect two random neurons
					int from = random(neurons.size());
					int to = random(neurons.size());
					Neuron.Connection connection = neurons.get(from).project(
							neurons.get(to), null);
					connectionList.add(connection);
				}

				for (int j = 0; j < gates; j++) {
					// pick a random gater neuron
					int gater = random(neurons.size());
					// pick a random connection to gate
					int connection = random(connectionList.size());
					// let the gater gate the connection
					neurons.get(gater).gate(connectionList.get(connection));
				}

				// connect the layers
				inputLayer.project(hiddenLayer);
				hiddenLayer.project(outputLayer);

				// set the layers of the network
				this.set(inputLayer, Arrays.asList(hiddenLayer), outputLayer);

				// trainer
				this.trainer = new Trainer(this);
			}
		}
	}// end architect
	
	public static interface Function<RET> {
		RET call();
	}

	public static interface Function1<A1, RET> {
		RET call(A1 arg);
	}

	public static interface Function2<A1, A2, RET> {
		RET call(A1 arg, A2 arg2);
	}

	public static interface CustomLog {
	}

	public static long now() {
		return System.currentTimeMillis();
	}

	public final static void log(Object... msg) {
		String s = "";
		for (Object o : msg)
			s += o + "";
		System.out.println(s);
	}

	public static int random(double v) {
		return (int) Math.floor(Math.random() * v);
	}
}// end Synaptic