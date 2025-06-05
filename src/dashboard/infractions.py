# Add to your Streamlit dashboard
def display_bucket_infractions():
    st.subheader("🪣 Bucket/Large Object Infractions")
    
    infractions = bucket_detector.get_active_infractions()
    
    if infractions:
        for infraction in infractions:
            severity_icon = "🔴" if infraction.severity == "critical" else "🟡"
            
            with st.expander(f"{severity_icon} {infraction.object_class} - {infraction.zone}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Entry Count", infraction.entry_count)
                    st.metric("Zone", infraction.zone.replace('_', ' ').title())
                    st.write(f"**Time:** {infraction.timestamp.strftime('%H:%M:%S')}")
                
                with col2:
                    if st.button(f"Clear Infraction", key=f"clear_{infraction.track_id}"):
                        bucket_detector.clear_infraction(infraction.track_id)
                        st.rerun()
    else:
        st.success("No active bucket infractions")