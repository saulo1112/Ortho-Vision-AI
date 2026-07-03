import { StyleSheet, Text } from 'react-native';

import { colors, fonts } from '../theme/tokens';

export function Disclaimer() {
  return (
    <Text style={styles.text}>
      Educational demo — not a medical device.{'\n'}Not intended for clinical use.
    </Text>
  );
}

const styles = StyleSheet.create({
  text: {
    color: colors.textTertiary,
    fontSize: 11,
    lineHeight: 16,
    textAlign: 'center',
    fontFamily: fonts?.sans,
  },
});
