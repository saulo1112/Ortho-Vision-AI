import { Redirect, router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { DetectionCard } from '../components/DetectionCard';
import { Disclaimer } from '../components/Disclaimer';
import { ChevronLeftIcon } from '../components/icons';
import { MaskOverlay } from '../components/MaskOverlay';
import { getAnalysis } from '../lib/analysis-store';
import { classStyle } from '../theme/classes';
import { colors, fonts, radius, spacing } from '../theme/tokens';

export default function ResultScreen() {
  const { width: windowWidth } = useWindowDimensions();
  const analysis = getAnalysis();
  if (!analysis) return <Redirect href="/" />;

  const { imageUri, response } = analysis;
  const aspect = response.image.height / response.image.width;
  const renderWidth = windowWidth - spacing.lg * 2;
  const renderHeight = renderWidth * aspect;

  return (
    <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.headerRow}>
          <Pressable
            onPress={() => router.back()}
            style={({ pressed }) => [styles.iconButton, pressed && styles.pressed]}
            hitSlop={8}
          >
            <ChevronLeftIcon color={colors.textSecondary} />
          </Pressable>
          <Text style={styles.title}>Result</Text>
          <View style={styles.headerSpacer} />
        </View>

        <MaskOverlay
          width={renderWidth}
          height={renderHeight}
          detections={response.detections}
          imageUri={imageUri}
        />

        <View style={styles.statsRow}>
          <Stat label="Detections" value={String(response.detections.length)} />
          <Stat label="Server time" value={`${response.timing_ms.total} ms`} />
          <Stat label="Model" value="v8s-seg" />
        </View>

        {response.detections.length > 0 ? (
          <View style={styles.list}>
            {response.detections.map((det) => (
              <DetectionCard key={det.id} detection={det} />
            ))}
          </View>
        ) : (
          <View style={styles.emptyBox}>
            <Text style={styles.emptyTitle}>No implants detected</Text>
            <Text style={styles.emptyCopy}>
              The model found no orthopedic implants above the 50% confidence threshold. Make
              sure the image is an X-ray showing the implant clearly.
            </Text>
          </View>
        )}

        {response.detections.length > 0 && (
          <View style={styles.legend}>
            {[...new Set(response.detections.map((d) => d.class_name))].map((name) => (
              <View key={name} style={styles.legendItem}>
                <View style={[styles.legendDot, { backgroundColor: classStyle(name).color }]} />
                <Text style={styles.legendLabel}>{classStyle(name).label}</Text>
              </View>
            ))}
          </View>
        )}

        <Pressable
          onPress={() => router.back()}
          style={({ pressed }) => [styles.newButton, pressed && styles.pressed]}
        >
          <Text style={styles.newButtonLabel}>New analysis</Text>
        </Pressable>

        <Disclaimer />
      </ScrollView>
    </SafeAreaView>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { padding: spacing.lg, gap: spacing.md },
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  title: { color: colors.text, fontSize: 18, fontWeight: '700', fontFamily: fonts?.sans },
  iconButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerSpacer: { width: 42, height: 42 },
  pressed: { opacity: 0.65 },
  statsRow: { flexDirection: 'row', gap: spacing.sm },
  stat: {
    flex: 1,
    padding: spacing.md,
    borderRadius: radius.control,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
    alignItems: 'center',
    gap: 2,
  },
  statValue: { color: colors.text, fontSize: 16, fontWeight: '700', fontFamily: fonts?.rounded },
  statLabel: { color: colors.textTertiary, fontSize: 11, fontFamily: fonts?.sans },
  list: { gap: spacing.sm },
  emptyBox: {
    padding: spacing.lg,
    borderRadius: radius.card,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
    gap: spacing.sm,
    alignItems: 'center',
  },
  emptyTitle: { color: colors.text, fontSize: 16, fontWeight: '600', fontFamily: fonts?.sans },
  emptyCopy: {
    color: colors.textSecondary,
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
    fontFamily: fonts?.sans,
  },
  legend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
    justifyContent: 'center',
  },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendDot: { width: 8, height: 8, borderRadius: 4 },
  legendLabel: { color: colors.textSecondary, fontSize: 12, fontFamily: fonts?.sans },
  newButton: {
    height: 50,
    borderRadius: radius.control,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  newButtonLabel: { color: '#06251F', fontSize: 15, fontWeight: '700', fontFamily: fonts?.sans },
});
