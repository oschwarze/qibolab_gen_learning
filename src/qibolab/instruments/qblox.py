from pulsar_qcm.pulsar_qcm import pulsar_qcm
from pulsar_qrm.pulsar_qrm import pulsar_qrm


class PulsarQRM(pulsar_qrm):
    """Class for interfacing with Pulsar QRM."""

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0, sync_en=True,
                 hardware_avg_en=True, acq_trigger_mode="sequencer",
                 debugging=False):
        # Instantiate base object from qblox library and connect to it
        super().__init__(label, ip)

        # Reset and configure
        self.reset()
        self.reference_source(ref_clock)
        self.scope_acq_sequencer_select(sequencer)
        self.scope_acq_avg_mode_en_path0(hardware_avg_en)
        self.scope_acq_avg_mode_en_path1(hardware_avg_en)
        self.scope_acq_trigger_mode_path0(acq_trigger_mode)
        self.scope_acq_trigger_mode_path1(acq_trigger_mode)

        self.sequencer = sequencer
        if self.sequencer == 1:
            self.sequencer1_sync_en(sync_en)
        else:
            self.sequencer0_sync_en(sync_en)

        self.debugging = debugging

    def setup(self, gain):
        if self.sequencer == 1:
            self.sequencer1_gain_awg_path0(gain)
            self.sequencer1_gain_awg_path1(gain)
        else:
            self.sequencer0_gain_awg_path0(gain)
            self.sequencer0_gain_awg_path1(gain)

    def translate(self, pulses):
        waveform = pulses[0].waveform()
        waveforms = {
            "modI_qrm": {"data": [], "index": 0},
            "modQ_qrm": {"data": [], "index": 1}
        }
        waveforms["modI_qrm"]["data"] = waveform.get("modI").get("data")
        waveforms["modQ_qrm"]["data"] = waveform.get("modQ").get("data")

        for pulse in pulses[1:]:
            waveform = pulse.waveform()
            waveforms["modI_qrm"]["data"] = np.concatenate((waveforms["modI_qrm"]["data"], np.zeros(4), waveform["modI"]["data"]))
            waveforms["modQ_qrm"]["data"] = np.concatenate((waveforms["modQ_qrm"]["data"], np.zeros(4), waveform["modQ"]["data"]))

        if self.debugging:
            # Plot the result
            fig, ax = plt.subplots(1, 1, figsize=(15, 15/2/1.61))
            ax.plot(combined_waveforms["modI_qrm"]["data"],'-',color='C0')
            ax.plot(combined_waveforms["modQ_qrm"]["data"],'-',color='C1')
            ax.title.set_text('Combined Pulses')

        return waveforms


class PulsarQCM(pulsar_qcm):

    def __init__(self, label, ip,
                 ref_clock="external", sequencer=0, sync_en=True):
        # Instantiate base object from qblox library and connect to it
        super().__init__(label, ip)

        # Reset and configure
        self.reset()
        self.reference_source(ref_clock)

        self.sequencer = sequencer
        if self.sequencer == 1:
            self.sequencer1_sync_en(sync_en)
        else:
            self.sequencer0_sync_en(sync_en)

    def setup(self, gain=0.5):
        if self.sequencer == 1:
            self.sequencer1_gain_awg_path0(gain)
            self.sequencer1_gain_awg_path1(gain)
        else:
            self.sequencer0_gain_awg_path0(gain)
            self.sequencer0_gain_awg_path1(gain)
